#include <finite_element/finite_element_extra_constitution.h>
#include <uipc/builtin/attribute_name.h>
#include <finite_element/constitutions/aerodynamic_damping_function.h>
#include <utils/make_spd.h>
#include <utils/matrix_assembler.h>
#include <map>

namespace uipc::backend::cuda
{
class AerodynamicDamping final : public FiniteElementExtraConstitution
{
    static constexpr U64   AerodynamicDampingUID = 33;
    static constexpr SizeT StencilSize           = 3;
    static constexpr SizeT HalfHessianSize = StencilSize * (StencilSize + 1) / 2;  // 6
    using Base = FiniteElementExtraConstitution;

  public:
    using Base::Base;
    U64 get_uid() const noexcept override { return AerodynamicDampingUID; }

    // Per-triangle data
    vector<Vector3i> h_stencils;
    vector<Float>    h_drag_coefficients;

    muda::DeviceBuffer<Vector3i> stencils;
    muda::DeviceBuffer<Float>    drag_coefficients;

    // Curvature-aware drag data (includes both drag reduction and inflation)
    bool          m_has_curvature = false;
    vector<Float> h_curvature_scales;
    vector<Float> h_inflate_scales;
    muda::DeviceBuffer<Float> curvature_scales;
    muda::DeviceBuffer<Float> inflate_scales;

    // Per-vertex adjacency (CSR): vertex -> list of adjacent triangle indices
    muda::DeviceBuffer<IndexT> unique_verts;
    muda::DeviceBuffer<IndexT> vert_adj_offsets;
    muda::DeviceBuffer<IndexT> vert_adj_tri_indices;

    // Per-vertex boundary flag (1 = boundary, 0 = interior; indexed by unique_verts order)
    muda::DeviceBuffer<IndexT> vert_is_boundary;

    // Per-vertex curvature output (indexed by global FEM vertex index)
    // vertex_H_v: clamped to ≤ 0 (for drag reduction on convex-facing-velocity regions)
    // vertex_H_v_pos: clamped to [0, 1] (for inflate on concave-facing-velocity regions)
    SizeT                     m_H_v_buffer_size = 0;
    muda::DeviceBuffer<Float> vertex_H_v;
    muda::DeviceBuffer<Float> vertex_H_v_pos;

    virtual void do_build(BuildInfo& info) override {}

    virtual void do_init(FilteredInfo& info) override
    {
        using ForEachInfo = FiniteElementMethod::ForEachInfo;
        auto geo_slots    = world().scene().geometries();

        list<Vector3i> stencil_list;
        list<Float>    drag_coeff_list;
        list<Float>    curv_scale_list;
        list<Float>    inflate_scale_list;

        info.for_each(
            geo_slots,
            [&](const ForEachInfo& I, geometry::SimplicialComplex& sc)
            {
                auto vertex_offset =
                    sc.meta().find<IndexT>(builtin::backend_fem_vertex_offset);
                UIPC_ASSERT(vertex_offset, "Vertex offset not found");
                auto vertex_offset_v = vertex_offset->view().front();

                auto triangles = sc.triangles().topo().view();
                auto dc        = sc.triangles().find<Float>("drag_coefficient");
                UIPC_ASSERT(dc, "drag_coefficient not found");
                auto dc_view = dc->view();

                auto cs = sc.triangles().find<Float>("curvature_scale");
                auto is = sc.triangles().find<Float>("inflate_scale");

                for(auto&& [i, tri] : enumerate(triangles))
                {
                    Vector3i stencil = tri;
                    stencil_list.push_back(stencil.array() + vertex_offset_v);
                    drag_coeff_list.push_back(dc_view[i]);
                    curv_scale_list.push_back(cs ? cs->view()[i] : 0.0);
                    inflate_scale_list.push_back(is ? is->view()[i] : 0.0);
                }
            });

        h_stencils.assign(stencil_list.begin(), stencil_list.end());
        h_drag_coefficients.assign(drag_coeff_list.begin(), drag_coeff_list.end());
        h_curvature_scales.assign(curv_scale_list.begin(), curv_scale_list.end());
        h_inflate_scales.assign(inflate_scale_list.begin(), inflate_scale_list.end());

        // Copy base data to GPU
        stencils.resize(h_stencils.size());
        stencils.view().copy_from(h_stencils.data());

        drag_coefficients.resize(h_drag_coefficients.size());
        drag_coefficients.view().copy_from(h_drag_coefficients.data());

        // Check if any curvature feature is active (curvature_scale or inflate_scale)
        for(SizeT i = 0; i < h_curvature_scales.size(); i++)
        {
            if(h_curvature_scales[i] != 0.0 || h_inflate_scales[i] != 0.0)
            {
                m_has_curvature = true;
                break;
            }
        }

        if(m_has_curvature)
        {
            // Copy curvature and inflate scales to GPU
            curvature_scales.resize(h_curvature_scales.size());
            curvature_scales.view().copy_from(h_curvature_scales.data());

            inflate_scales.resize(h_inflate_scales.size());
            inflate_scales.view().copy_from(h_inflate_scales.data());

            // Build per-vertex adjacency (CSR)
            set<IndexT> vert_set;
            for(auto& s : h_stencils)
            {
                vert_set.insert(s[0]);
                vert_set.insert(s[1]);
                vert_set.insert(s[2]);
            }

            vector<IndexT> h_unique_verts(vert_set.begin(), vert_set.end());

            // Build adjacency map: global vert index -> list of triangle indices
            unordered_map<IndexT, vector<IndexT>> adj;
            for(SizeT ti = 0; ti < h_stencils.size(); ti++)
            {
                for(int k = 0; k < 3; k++)
                    adj[h_stencils[ti][k]].push_back(ti);
            }

            // Flatten into CSR
            vector<IndexT> h_offsets(h_unique_verts.size() + 1);
            vector<IndexT> h_adj_tris;
            h_offsets[0] = 0;
            for(SizeT i = 0; i < h_unique_verts.size(); i++)
            {
                auto& tris = adj[h_unique_verts[i]];
                h_adj_tris.insert(h_adj_tris.end(), tris.begin(), tris.end());
                h_offsets[i + 1] = h_adj_tris.size();
            }

            // Detect boundary vertices: edge shared by only 1 triangle
            std::map<std::pair<IndexT, IndexT>, SizeT> edge_tri_count;
            for(SizeT si = 0; si < h_stencils.size(); si++)
            {
                auto& s = h_stencils[si];
                for(int k = 0; k < 3; k++)
                {
                    IndexT a = s[k], b = s[(k + 1) % 3];
                    if(a > b) std::swap(a, b);
                    edge_tri_count[std::make_pair(a, b)]++;
                }
            }
            set<IndexT> boundary_verts;
            for(auto it = edge_tri_count.begin(); it != edge_tri_count.end(); ++it)
            {
                if(it->second == 1)
                {
                    boundary_verts.insert(it->first.first);
                    boundary_verts.insert(it->first.second);
                }
            }

            // Build per-unique-vert boundary flag (same order as h_unique_verts)
            vector<IndexT> h_is_boundary(h_unique_verts.size());
            for(SizeT i = 0; i < h_unique_verts.size(); i++)
                h_is_boundary[i] = boundary_verts.count(h_unique_verts[i]) ? 1 : 0;

            // H_v buffer sized by max global vertex index + 1
            m_H_v_buffer_size = *vert_set.rbegin() + 1;

            // Copy to GPU
            unique_verts.resize(h_unique_verts.size());
            unique_verts.view().copy_from(h_unique_verts.data());

            vert_adj_offsets.resize(h_offsets.size());
            vert_adj_offsets.view().copy_from(h_offsets.data());

            vert_adj_tri_indices.resize(h_adj_tris.size());
            vert_adj_tri_indices.view().copy_from(h_adj_tris.data());

            vert_is_boundary.resize(h_is_boundary.size());
            vert_is_boundary.view().copy_from(h_is_boundary.data());

            vertex_H_v.resize(m_H_v_buffer_size);
            vertex_H_v_pos.resize(m_H_v_buffer_size);
        }
    }

    virtual void do_report_extent(ReportExtentInfo& info) override
    {
        info.energy_count(stencils.size());
        info.gradient_count(stencils.size() * StencilSize);

        if(info.gradient_only())
            return;

        info.hessian_count(stencils.size() * HalfHessianSize);
    }

    // Launch curvature computation kernels.
    // Computes per-vertex mean curvature projected onto velocity direction
    // using the cotangent Laplacian: H_vec = (1/2A) Σ (cot α + cot β)(xj - xi)
    // Then:
    //   H_v     = min(dot(H_vec, v_hat), 0)  — drag reduction (convex facing velocity)
    //   H_v_pos = clamp(dot(H_vec, v_hat), 0, 1) — inflate (concave facing velocity)
    //
    // H_vec points toward center of curvature (cotangent Laplacian).
    // When convex side faces velocity (dome), they oppose → H_v < 0 → reduced drag.
    // When concave side faces velocity (parachute), same direction → H_v_pos > 0 → inflate.
#define LAUNCH_CURVATURE_KERNELS(xs_expr, x_prevs_expr)                        \
    do                                                                         \
    {                                                                          \
        ParallelFor()                                                          \
            .file_line(__FILE__, __LINE__)                                      \
            .apply(unique_verts.size(),                                         \
                   [uv_  = unique_verts.viewer().name("unique_verts"),          \
                    hv_  = vertex_H_v.viewer().name("H_v_zero"),               \
                    hvp_ = vertex_H_v_pos.viewer().name("H_v_pos_zero")]       \
                       __device__(int I)                                        \
                   {                                                            \
                       hv_(uv_(I)) = 0.0;                                      \
                       hvp_(uv_(I)) = 0.0;                                     \
                   });                                                          \
                                                                               \
        ParallelFor()                                                          \
            .file_line(__FILE__, __LINE__)                                      \
            .apply(                                                            \
                unique_verts.size(),                                            \
                [uv_      = unique_verts.viewer().name("unique_verts"),         \
                 bnd_     = vert_is_boundary.viewer().name("boundary"),         \
                 offsets_  = vert_adj_offsets.viewer().name("adj_offsets"),      \
                 adj_      = vert_adj_tri_indices.viewer().name("adj_tris"),    \
                 sten_     = stencils.viewer().name("stencils"),                \
                 xs_       = xs_expr,                                           \
                 xp_       = x_prevs_expr,                                     \
                 hv_       = vertex_H_v.viewer().name("H_v"),                  \
                 hvp_      = vertex_H_v_pos.viewer().name("H_v_pos")]          \
                    __device__(int I) {                                         \
                    if(bnd_(I)) return;                                         \
                    IndexT vi    = uv_(I);                                      \
                    IndexT start = offsets_(I);                                 \
                    IndexT end   = offsets_(I + 1);                             \
                                                                               \
                    Vector3 xi       = xs_(vi);                                 \
                    Vector3 H_vec    = Vector3::Zero();                         \
                    Float   area_sum = 0.0;                                     \
                                                                               \
                    for(IndexT t = start; t < end; t++)                         \
                    {                                                           \
                        IndexT   ti  = adj_(t);                                 \
                        Vector3i stl = sten_(ti);                               \
                                                                               \
                        int li = 0;                                             \
                        if(stl[1] == vi) li = 1;                                \
                        else if(stl[2] == vi) li = 2;                           \
                                                                               \
                        int jl = (li + 1) % 3;                                  \
                        int kl = (li + 2) % 3;                                  \
                                                                               \
                        Vector3 xj = xs_(stl[jl]);                              \
                        Vector3 xk = xs_(stl[kl]);                              \
                                                                               \
                        Vector3 ec  = (xj - xi).cross(xk - xi);                \
                        Float   cn  = ec.norm();                                \
                        area_sum += cn / 6.0;                                   \
                                                                               \
                        if(cn < 1e-20) continue;                                \
                                                                               \
                        Vector3 eki = xi - xk;                                  \
                        Vector3 ekj = xj - xk;                                  \
                        Float   ck  = eki.cross(ekj).norm();                    \
                        Float   ctk = (ck > 1e-20)                              \
                                          ? eki.dot(ekj) / ck                   \
                                          : 0.0;                                \
                                                                               \
                        Vector3 eji = xi - xj;                                  \
                        Vector3 ejk = xk - xj;                                  \
                        Float   cj  = eji.cross(ejk).norm();                    \
                        Float   ctj = (cj > 1e-20)                              \
                                          ? eji.dot(ejk) / cj                   \
                                          : 0.0;                                \
                                                                               \
                        H_vec += ctk * (xj - xi) + ctj * (xk - xi);            \
                    }                                                           \
                                                                               \
                    if(area_sum > 1e-20)                                        \
                        H_vec /= (2.0 * area_sum);                             \
                                                                               \
                    Vector3 disp = xi - xp_(vi);                                \
                    Float   dn   = disp.norm();                                 \
                                                                               \
                    Float hval = 0.0;                                            \
                    if(dn > 1e-10)                                              \
                        hval = H_vec.dot(disp / dn);                            \
                                                                               \
                    hv_(vi) = min(hval, (Float)0.0);                             \
                    hvp_(vi) = max(min(hval, (Float)1.0), (Float)0.0);          \
                });                                                             \
    } while(0)

    virtual void do_compute_energy(ComputeEnergyInfo& info) override
    {
        using namespace muda;
        namespace AD = sym::aerodynamic_damping;

        if(m_has_curvature)
        {
            LAUNCH_CURVATURE_KERNELS(info.xs().viewer().name("xs"),
                                     info.x_prevs().viewer().name("x_prevs"));

            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(info.energies().size(),
                       [stencils    = stencils.viewer().name("stencils"),
                        drag_coeffs = drag_coefficients.viewer().name("drag_coefficients"),
                        curv_scales = curvature_scales.viewer().name("curvature_scales"),
                        infl_scales = inflate_scales.viewer().name("inflate_scales"),
                        H_v         = vertex_H_v.viewer().name("H_v"),
                        H_v_pos     = vertex_H_v_pos.viewer().name("H_v_pos"),
                        xs          = info.xs().viewer().name("xs"),
                        x_prevs     = info.x_prevs().viewer().name("x_prevs"),
                        energies    = info.energies().viewer().name("energies"),
                        dt          = info.dt()] __device__(int I)
                       {
                           Vector3i stencil = stencils(I);
                           Float    coeff   = drag_coeffs(I);

                           // Curvature drag reduction: reduce drag on convex-facing-velocity regions
                           Float cs = curv_scales(I);
                           if(cs != 0.0)
                           {
                               Float H_v_avg = (H_v(stencil[0]) + H_v(stencil[1])
                                                + H_v(stencil[2]))
                                               / 3.0;
                               coeff *= max(1.0 + cs * H_v_avg, (Float)0.0);
                           }

                           // Curvature inflate: increase drag on concave-facing-velocity regions
                           Float is = infl_scales(I);
                           if(is != 0.0)
                           {
                               Float H_v_pos_avg =
                                   (H_v_pos(stencil[0]) + H_v_pos(stencil[1])
                                    + H_v_pos(stencil[2]))
                                   / 3.0;
                               coeff *= (1.0 + is * H_v_pos_avg);
                           }

                           Vector3 x0 = xs(stencil[0]);
                           Vector3 x1 = xs(stencil[1]);
                           Vector3 x2 = xs(stencil[2]);

                           Vector3 x0p = x_prevs(stencil[0]);
                           Vector3 x1p = x_prevs(stencil[1]);
                           Vector3 x2p = x_prevs(stencil[2]);

                           energies(I) = AD::E(x0, x1, x2, x0p, x1p, x2p, coeff);
                       });
        }
        else
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(info.energies().size(),
                       [stencils    = stencils.viewer().name("stencils"),
                        drag_coeffs = drag_coefficients.viewer().name("drag_coefficients"),
                        xs          = info.xs().viewer().name("xs"),
                        x_prevs     = info.x_prevs().viewer().name("x_prevs"),
                        energies    = info.energies().viewer().name("energies"),
                        dt          = info.dt()] __device__(int I)
                       {
                           Vector3i stencil = stencils(I);
                           Float    coeff   = drag_coeffs(I);

                           Vector3 x0 = xs(stencil[0]);
                           Vector3 x1 = xs(stencil[1]);
                           Vector3 x2 = xs(stencil[2]);

                           Vector3 x0p = x_prevs(stencil[0]);
                           Vector3 x1p = x_prevs(stencil[1]);
                           Vector3 x2p = x_prevs(stencil[2]);

                           energies(I) = AD::E(x0, x1, x2, x0p, x1p, x2p, coeff);
                       });
        }
    }

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        using namespace muda;
        namespace AD = sym::aerodynamic_damping;

        if(m_has_curvature)
        {
            LAUNCH_CURVATURE_KERNELS(info.xs().viewer().name("xs"),
                                     info.x_prevs().viewer().name("x_prevs"));

            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(stencils.size(),
                       [stencils      = stencils.viewer().name("stencils"),
                        drag_coeffs   = drag_coefficients.viewer().name("drag_coefficients"),
                        curv_scales   = curvature_scales.viewer().name("curvature_scales"),
                        infl_scales   = inflate_scales.viewer().name("inflate_scales"),
                        H_v           = vertex_H_v.viewer().name("H_v"),
                        H_v_pos       = vertex_H_v_pos.viewer().name("H_v_pos"),
                        xs            = info.xs().viewer().name("xs"),
                        x_prevs       = info.x_prevs().viewer().name("x_prevs"),
                        G3s           = info.gradients().viewer().name("G3s"),
                        H3x3s        = info.hessians().viewer().name("H3x3s"),
                        gradient_only = info.gradient_only(),
                        dt            = info.dt()] __device__(int I)
                       {
                           Vector3i stencil = stencils(I);
                           Float    coeff   = drag_coeffs(I);

                           // Curvature drag reduction
                           Float cs = curv_scales(I);
                           if(cs != 0.0)
                           {
                               Float H_v_avg = (H_v(stencil[0]) + H_v(stencil[1])
                                                + H_v(stencil[2]))
                                               / 3.0;
                               coeff *= max(1.0 + cs * H_v_avg, (Float)0.0);
                           }

                           // Curvature inflate
                           Float is = infl_scales(I);
                           if(is != 0.0)
                           {
                               Float H_v_pos_avg =
                                   (H_v_pos(stencil[0]) + H_v_pos(stencil[1])
                                    + H_v_pos(stencil[2]))
                                   / 3.0;
                               coeff *= (1.0 + is * H_v_pos_avg);
                           }

                           Vector3 x0 = xs(stencil[0]);
                           Vector3 x1 = xs(stencil[1]);
                           Vector3 x2 = xs(stencil[2]);

                           Vector3 x0p = x_prevs(stencil[0]);
                           Vector3 x1p = x_prevs(stencil[1]);
                           Vector3 x2p = x_prevs(stencil[2]);

                           // Gradient (9-vector)
                           Vector<Float, 9> G;
                           AD::dEdx(G, x0, x1, x2, x0p, x1p, x2p, coeff);

                           // Assemble gradient
                           DoubletVectorAssembler DVA{G3s};
                           DVA.segment<StencilSize>(I * StencilSize).write(stencil, G);

                           if(gradient_only)
                               return;

                           // Hessian (9x9 matrix, Gauss-Newton PSD approximation)
                           Matrix<Float, 9, 9> H;
                           AD::ddEddx(H, x0, x1, x2, x0p, x1p, x2p, coeff);

                           // Assemble Hessian (PSD by construction — Gauss-Newton outer product)
                           TripletMatrixAssembler TMA{H3x3s};
                           TMA.half_block<StencilSize>(I * HalfHessianSize)
                               .write(stencil, H);
                       });
        }
        else
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(stencils.size(),
                       [stencils      = stencils.viewer().name("stencils"),
                        drag_coeffs   = drag_coefficients.viewer().name("drag_coefficients"),
                        xs            = info.xs().viewer().name("xs"),
                        x_prevs       = info.x_prevs().viewer().name("x_prevs"),
                        G3s           = info.gradients().viewer().name("G3s"),
                        H3x3s        = info.hessians().viewer().name("H3x3s"),
                        gradient_only = info.gradient_only(),
                        dt            = info.dt()] __device__(int I)
                       {
                           Vector3i stencil = stencils(I);
                           Float    coeff   = drag_coeffs(I);

                           Vector3 x0 = xs(stencil[0]);
                           Vector3 x1 = xs(stencil[1]);
                           Vector3 x2 = xs(stencil[2]);

                           Vector3 x0p = x_prevs(stencil[0]);
                           Vector3 x1p = x_prevs(stencil[1]);
                           Vector3 x2p = x_prevs(stencil[2]);

                           // Gradient (9-vector)
                           Vector<Float, 9> G;
                           AD::dEdx(G, x0, x1, x2, x0p, x1p, x2p, coeff);

                           // Assemble gradient
                           DoubletVectorAssembler DVA{G3s};
                           DVA.segment<StencilSize>(I * StencilSize).write(stencil, G);

                           if(gradient_only)
                               return;

                           // Hessian (9x9 matrix, Gauss-Newton PSD approximation)
                           Matrix<Float, 9, 9> H;
                           AD::ddEddx(H, x0, x1, x2, x0p, x1p, x2p, coeff);

                           // Assemble Hessian (PSD by construction — Gauss-Newton outer product)
                           TripletMatrixAssembler TMA{H3x3s};
                           TMA.half_block<StencilSize>(I * HalfHessianSize)
                               .write(stencil, H);
                       });
        }
    }

#undef LAUNCH_CURVATURE_KERNELS
};

REGISTER_SIM_SYSTEM(AerodynamicDamping);
}  // namespace uipc::backend::cuda
