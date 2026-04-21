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

    // Curvature-aware drag data
    bool          m_has_curvature = false;
    vector<Float> h_curvature_scales;
    muda::DeviceBuffer<Float> curvature_scales;

    // Inflate data
    bool          m_has_inflate = false;
    vector<Float> h_inflate_scales;
    muda::DeviceBuffer<Float> inflate_scales;

    // Per-vertex adjacency (CSR): vertex -> list of adjacent triangle indices
    muda::DeviceBuffer<IndexT> unique_verts;
    muda::DeviceBuffer<IndexT> vert_adj_offsets;
    muda::DeviceBuffer<IndexT> vert_adj_tri_indices;

    // Per-vertex boundary flag
    muda::DeviceBuffer<IndexT> vert_is_boundary;

    // Per-vertex curvature (for drag reduction)
    SizeT                     m_H_v_buffer_size = 0;
    muda::DeviceBuffer<Float> vertex_H_v;

    // Per-vertex H_vec (cotangent Laplacian mean curvature vector, for inflate direction)
    muda::DeviceBuffer<Vector3> vertex_H_vec;

    // Per-triangle entity index, per-vertex entity index
    muda::DeviceBuffer<IndexT> tri_entity_idx;
    muda::DeviceBuffer<IndexT> vert_entity_idx;

    // Per-entity H_vg
    SizeT                      m_entity_count = 0;
    muda::DeviceBuffer<Float>  entity_H_vg_sum;
    muda::DeviceBuffer<IndexT> entity_vert_count;
    muda::DeviceBuffer<Float>  entity_H_vg;

    // Per-entity net inflate force sum (for zero-net-force subtraction)
    muda::DeviceBuffer<Vector3> entity_net_force;
    muda::DeviceBuffer<IndexT>  entity_tri_count;

    // Per-triangle inflate force vector (frozen for Newton solve)
    // F_inflate applied as constant force to each vertex of the triangle
    muda::DeviceBuffer<Vector3> tri_inflate_force;

    // Inflate forces only depend on x_prevs (constant during Newton solve).
    // Computed once in the first do_compute_energy call; the cached buffer
    // is reused by subsequent energy calls (line search) and
    // do_compute_gradient_hessian.
    bool m_inflate_forces_computed = false;

    virtual void do_build(BuildInfo& info) override {}

    virtual void do_init(FilteredInfo& info) override
    {
        using ForEachInfo = FiniteElementMethod::ForEachInfo;
        auto geo_slots    = world().scene().geometries();

        list<Vector3i> stencil_list;
        list<Float>    drag_coeff_list;
        list<Float>    curv_scale_list;
        list<Float>    inflate_scale_list;
        list<IndexT>   tri_entity_list;

        unordered_map<IndexT, IndexT> vert_to_entity;
        IndexT entity_idx = 0;

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

                auto verts = sc.positions().view();
                for(SizeT vi = 0; vi < verts.size(); vi++)
                    vert_to_entity[vi + vertex_offset_v] = entity_idx;

                for(auto&& [i, tri] : enumerate(triangles))
                {
                    Vector3i stencil = tri;
                    stencil_list.push_back(stencil.array() + vertex_offset_v);
                    drag_coeff_list.push_back(dc_view[i]);
                    curv_scale_list.push_back(cs ? cs->view()[i] : 0.0);
                    inflate_scale_list.push_back(is ? is->view()[i] : 0.0);
                    tri_entity_list.push_back(entity_idx);
                }

                entity_idx++;
            });

        m_entity_count = entity_idx;

        h_stencils.assign(stencil_list.begin(), stencil_list.end());
        h_drag_coefficients.assign(drag_coeff_list.begin(), drag_coeff_list.end());
        h_curvature_scales.assign(curv_scale_list.begin(), curv_scale_list.end());
        h_inflate_scales.assign(inflate_scale_list.begin(), inflate_scale_list.end());

        stencils.resize(h_stencils.size());
        stencils.view().copy_from(h_stencils.data());

        drag_coefficients.resize(h_drag_coefficients.size());
        drag_coefficients.view().copy_from(h_drag_coefficients.data());

        for(SizeT i = 0; i < h_curvature_scales.size(); i++)
            if(h_curvature_scales[i] != 0.0) { m_has_curvature = true; break; }

        for(SizeT i = 0; i < h_inflate_scales.size(); i++)
            if(h_inflate_scales[i] != 0.0) { m_has_inflate = true; break; }

        bool needs_adjacency = m_has_curvature || m_has_inflate;

        if(needs_adjacency)
        {
            curvature_scales.resize(h_curvature_scales.size());
            curvature_scales.view().copy_from(h_curvature_scales.data());

            inflate_scales.resize(h_inflate_scales.size());
            inflate_scales.view().copy_from(h_inflate_scales.data());

            // Build vertex adjacency CSR
            set<IndexT> vert_set;
            for(auto& s : h_stencils)
            {
                vert_set.insert(s[0]);
                vert_set.insert(s[1]);
                vert_set.insert(s[2]);
            }

            vector<IndexT> h_unique_verts(vert_set.begin(), vert_set.end());

            unordered_map<IndexT, vector<IndexT>> adj;
            for(SizeT ti = 0; ti < h_stencils.size(); ti++)
                for(int k = 0; k < 3; k++)
                    adj[h_stencils[ti][k]].push_back(ti);

            vector<IndexT> h_offsets(h_unique_verts.size() + 1);
            vector<IndexT> h_adj_tris;
            h_offsets[0] = 0;
            for(SizeT i = 0; i < h_unique_verts.size(); i++)
            {
                auto& tris = adj[h_unique_verts[i]];
                h_adj_tris.insert(h_adj_tris.end(), tris.begin(), tris.end());
                h_offsets[i + 1] = h_adj_tris.size();
            }

            // Boundary detection
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
                if(it->second == 1)
                {
                    boundary_verts.insert(it->first.first);
                    boundary_verts.insert(it->first.second);
                }

            vector<IndexT> h_is_boundary(h_unique_verts.size());
            for(SizeT i = 0; i < h_unique_verts.size(); i++)
                h_is_boundary[i] = boundary_verts.count(h_unique_verts[i]) ? 1 : 0;

            m_H_v_buffer_size = *vert_set.rbegin() + 1;

            // Upload vertex adjacency
            unique_verts.resize(h_unique_verts.size());
            unique_verts.view().copy_from(h_unique_verts.data());
            vert_adj_offsets.resize(h_offsets.size());
            vert_adj_offsets.view().copy_from(h_offsets.data());
            vert_adj_tri_indices.resize(h_adj_tris.size());
            vert_adj_tri_indices.view().copy_from(h_adj_tris.data());
            vert_is_boundary.resize(h_is_boundary.size());
            vert_is_boundary.view().copy_from(h_is_boundary.data());

            vertex_H_v.resize(m_H_v_buffer_size);

            if(m_has_inflate)
            {
                vertex_H_vec.resize(m_H_v_buffer_size);

                vector<IndexT> h_vert_entity(m_H_v_buffer_size, 0);
                for(auto& [gv, eidx] : vert_to_entity)
                    if((SizeT)gv < m_H_v_buffer_size)
                        h_vert_entity[gv] = eidx;

                vert_entity_idx.resize(m_H_v_buffer_size);
                vert_entity_idx.view().copy_from(h_vert_entity.data());

                vector<IndexT> h_tri_entity(tri_entity_list.begin(), tri_entity_list.end());
                tri_entity_idx.resize(h_tri_entity.size());
                tri_entity_idx.view().copy_from(h_tri_entity.data());

                entity_H_vg_sum.resize(m_entity_count);
                entity_vert_count.resize(m_entity_count);
                entity_H_vg.resize(m_entity_count);

                entity_net_force.resize(m_entity_count);
                entity_tri_count.resize(m_entity_count);

                tri_inflate_force.resize(h_stencils.size());
            }
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

    // Curvature drag kernel: compute per-vertex H_v from current xs
#define LAUNCH_CURVATURE_DRAG_KERNEL(xs_expr, x_prevs_expr)                    \
    do                                                                         \
    {                                                                          \
        ParallelFor()                                                          \
            .file_line(__FILE__, __LINE__)                                      \
            .apply(unique_verts.size(),                                         \
                   [uv_  = unique_verts.viewer().name("unique_verts"),          \
                    hv_  = vertex_H_v.viewer().name("H_v_zero")]               \
                       __device__(int I) { hv_(uv_(I)) = 0.0; });             \
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
                 hv_       = vertex_H_v.viewer().name("H_v")]                  \
                    __device__(int I) {                                         \
                    if(bnd_(I)) return;                                         \
                    IndexT vi    = uv_(I);                                      \
                    IndexT start = offsets_(I);                                 \
                    IndexT end   = offsets_(I + 1);                             \
                    Vector3 xi       = xs_(vi);                                 \
                    Vector3 H_vec    = Vector3::Zero();                         \
                    Float   area_sum = 0.0;                                     \
                    for(IndexT t = start; t < end; t++)                         \
                    {                                                           \
                        IndexT   ti  = adj_(t);                                 \
                        Vector3i stl = sten_(ti);                               \
                        int li = 0;                                             \
                        if(stl[1] == vi) li = 1;                                \
                        else if(stl[2] == vi) li = 2;                           \
                        int jl = (li + 1) % 3, kl = (li + 2) % 3;              \
                        Vector3 xj = xs_(stl[jl]);                              \
                        Vector3 xk = xs_(stl[kl]);                              \
                        Vector3 ec  = (xj - xi).cross(xk - xi);                \
                        Float   cn  = ec.norm();                                \
                        area_sum += cn / 6.0;                                   \
                        if(cn < 1e-20) continue;                                \
                        Vector3 eki = xi - xk, ekj = xj - xk;                  \
                        Float   ck  = eki.cross(ekj).norm();                    \
                        Float   ctk = (ck > 1e-20) ? eki.dot(ekj)/ck : 0.0;    \
                        Vector3 eji = xi - xj, ejk = xk - xj;                  \
                        Float   cj  = eji.cross(ejk).norm();                    \
                        Float   ctj = (cj > 1e-20) ? eji.dot(ejk)/cj : 0.0;    \
                        H_vec += ctk * (xj - xi) + ctj * (xk - xi);            \
                    }                                                           \
                    if(area_sum > 1e-20) H_vec /= (2.0 * area_sum);            \
                    Vector3 disp = xi - xp_(vi);                                \
                    Float   dn   = disp.norm();                                 \
                    Float hval = (dn > 1e-10) ? H_vec.dot(disp / dn) : 0.0;    \
                    hv_(vi) = min(hval, (Float)0.0);                            \
                });                                                             \
    } while(0)

    // Compute per-triangle inflate force vector from x_prevs (constant during Newton).
    // Inflate = constant force per triangle pushing outward on concave regions.
    //
    // Per triangle, the inflate force per vertex is:
    //   F_v = inflate_scale * min(H_vg, 1) * (area/3) * n_hat * sign
    //
    // Added as linear energy E_inflate = -F_v . x_i → constant gradient, zero Hessian.
    template <typename ViewerT>
    void compute_inflate_forces(ViewerT x_prevs_viewer)
    {
        using namespace muda;

        // 1. Zero H_vec and entity accumulators
        ParallelFor().file_line(__FILE__, __LINE__)
            .apply(unique_verts.size(),
                   [uv_ = unique_verts.viewer().name("uv"),
                    hvec_ = vertex_H_vec.viewer().name("hvec")]
                       __device__(int I) { hvec_(uv_(I)) = Vector3::Zero(); });

        ParallelFor().file_line(__FILE__, __LINE__)
            .apply(m_entity_count,
                   [hvg_sum_ = entity_H_vg_sum.viewer().name("hvg_sum"),
                    vcnt_ = entity_vert_count.viewer().name("vcnt"),
                    hvg_ = entity_H_vg.viewer().name("hvg")]
                       __device__(int I) { hvg_sum_(I) = 0.0; vcnt_(I) = 0; hvg_(I) = 0.0; });

        // 2. Compute H_vec from x_prevs, accumulate H_vg
        ParallelFor().file_line(__FILE__, __LINE__)
            .apply(unique_verts.size(),
                [uv_      = unique_verts.viewer().name("uv"),
                 bnd_     = vert_is_boundary.viewer().name("bnd"),
                 offsets_  = vert_adj_offsets.viewer().name("off"),
                 adj_      = vert_adj_tri_indices.viewer().name("adj"),
                 sten_     = stencils.viewer().name("sten"),
                 xp_       = x_prevs_viewer,
                 hvec_     = vertex_H_vec.viewer().name("hvec"),
                 vent_     = vert_entity_idx.viewer().name("vent"),
                 hvg_sum_  = entity_H_vg_sum.viewer().name("hvg_sum"),
                 vcnt_     = entity_vert_count.viewer().name("vcnt")]
                    __device__(int I) {
                    if(bnd_(I)) return;
                    IndexT vi = uv_(I);
                    IndexT start = offsets_(I), end = offsets_(I + 1);
                    Vector3 xi = xp_(vi);
                    Vector3 H_vec = Vector3::Zero();
                    Float area_sum = 0.0;
                    for(IndexT t = start; t < end; t++)
                    {
                        IndexT ti = adj_(t);
                        Vector3i stl = sten_(ti);
                        int li = 0;
                        if(stl[1] == vi) li = 1;
                        else if(stl[2] == vi) li = 2;
                        int jl = (li + 1) % 3, kl = (li + 2) % 3;
                        Vector3 xj = xp_(stl[jl]), xk = xp_(stl[kl]);
                        Vector3 ec = (xj - xi).cross(xk - xi);
                        Float cn = ec.norm();
                        area_sum += cn / 6.0;
                        if(cn < 1e-20) continue;
                        Vector3 eki = xi - xk, ekj = xj - xk;
                        Float ck = eki.cross(ekj).norm();
                        Float ctk = (ck > 1e-20) ? eki.dot(ekj)/ck : 0.0;
                        Vector3 eji = xi - xj, ejk = xk - xj;
                        Float cj = eji.cross(ejk).norm();
                        Float ctj = (cj > 1e-20) ? eji.dot(ejk)/cj : 0.0;
                        H_vec += ctk * (xj - xi) + ctj * (xk - xi);
                    }
                    if(area_sum > 1e-20) H_vec /= (2.0 * area_sum);
                    hvec_(vi) = H_vec;
                    Float H_mag = H_vec.norm();
                    IndexT eidx = vent_(vi);
                    atomicAdd(&hvg_sum_(eidx), H_mag);
                    atomicAdd(&vcnt_(eidx), 1);
                });

        // 3. Normalize H_vg
        ParallelFor().file_line(__FILE__, __LINE__)
            .apply(m_entity_count,
                   [hvg_sum_ = entity_H_vg_sum.viewer().name("hvg_sum"),
                    vcnt_ = entity_vert_count.viewer().name("vcnt"),
                    hvg_ = entity_H_vg.viewer().name("hvg")]
                       __device__(int I)
                   {
                       hvg_(I) = (vcnt_(I) > 0) ? hvg_sum_(I) / (Float)vcnt_(I) : 0.0;
                   });

        // 4. Compute raw per-triangle inflate force vector
        ParallelFor().file_line(__FILE__, __LINE__)
            .apply(m_entity_count,
                   [net_ = entity_net_force.viewer().name("net"),
                    cnt_ = entity_tri_count.viewer().name("cnt")]
                       __device__(int I) { net_(I) = Vector3::Zero(); cnt_(I) = 0; });

        ParallelFor().file_line(__FILE__, __LINE__)
            .apply(stencils.size(),
                   [sten_    = stencils.viewer().name("sten"),
                    infl_    = inflate_scales.viewer().name("infl"),
                    hvec_    = vertex_H_vec.viewer().name("hvec"),
                    tri_ent_ = tri_entity_idx.viewer().name("tri_ent"),
                    hvg_     = entity_H_vg.viewer().name("hvg"),
                    net_     = entity_net_force.viewer().name("net"),
                    cnt_     = entity_tri_count.viewer().name("cnt"),
                    xp_      = x_prevs_viewer,
                    out_     = tri_inflate_force.viewer().name("infl_force")]
                       __device__(int I)
                   {
                       Float is = infl_(I);
                       if(is == 0.0)
                       {
                           out_(I) = Vector3::Zero();
                           return;
                       }

                       Vector3i stencil = sten_(I);
                       Vector3 x0p = xp_(stencil[0]);
                       Vector3 x1p = xp_(stencil[1]);
                       Vector3 x2p = xp_(stencil[2]);

                       Vector3 c_prev = (x1p - x0p).cross(x2p - x0p);
                       Float cn = c_prev.norm();
                       if(cn < 1e-20)
                       {
                           out_(I) = Vector3::Zero();
                           return;
                       }
                       Vector3 n_hat = c_prev / cn;
                       Float area = cn / 2.0;

                       IndexT eidx = tri_ent_(I);
                       Float H_vg_val = hvg_(eidx);
                       Float H_vg_gate = min(H_vg_val, (Float)1.0);

                       // Sign: push outward on concave side
                       Vector3 H_vec_avg = (hvec_(stencil[0]) + hvec_(stencil[1])
                                            + hvec_(stencil[2])) / 3.0;
                       Float dot_Hc = H_vec_avg.dot(c_prev);
                       Float H_avg_n = H_vec_avg.norm();
                       Float norm_dot = (H_avg_n > 1e-20) ? dot_Hc / (cn * H_avg_n) : 0.0;
                       Float sign_val = -tanh(20.0 * norm_dot);

                       // Force per vertex (1/3 of total face force)
                       Float force_mag = is * H_vg_gate * area / 3.0 * sign_val;
                       Vector3 F = force_mag * n_hat;
                       out_(I) = F;

                       // Accumulate net force per entity (each tri contributes F to 3 vertices)
                       atomicAdd(&net_(eidx)(0), 3.0 * F(0));
                       atomicAdd(&net_(eidx)(1), 3.0 * F(1));
                       atomicAdd(&net_(eidx)(2), 3.0 * F(2));
                       atomicAdd(&cnt_(eidx), 1);
                   });

        // 5. Subtract net force to make inflate zero-net-force (pure shape change)
        // Each triangle applies F to 3 vertices. Total force = sum of 3*F over all tris.
        // Correction per vertex = -net / (3 * tri_count), applied as correction per tri.
        ParallelFor().file_line(__FILE__, __LINE__)
            .apply(stencils.size(),
                   [tri_ent_ = tri_entity_idx.viewer().name("tri_ent"),
                    net_     = entity_net_force.viewer().name("net"),
                    cnt_     = entity_tri_count.viewer().name("cnt"),
                    infl_    = inflate_scales.viewer().name("infl"),
                    out_     = tri_inflate_force.viewer().name("infl_force")]
                       __device__(int I)
                   {
                       if(infl_(I) == 0.0) return;
                       IndexT eidx = tri_ent_(I);
                       IndexT n = cnt_(eidx);
                       if(n == 0) return;
                       Vector3 correction = net_(eidx) / (3.0 * (Float)n);
                       out_(I) -= correction;
                   });
    }

    virtual void do_compute_energy(ComputeEnergyInfo& info) override
    {
        using namespace muda;
        namespace AD = sym::aerodynamic_damping;

        // Curvature drag
        if(m_has_curvature)
        {
            LAUNCH_CURVATURE_DRAG_KERNEL(info.xs().viewer().name("xs"),
                                         info.x_prevs().viewer().name("x_prevs"));
        }

        // Compute inflate forces on first energy call of this Newton step.
        // x_prevs is constant during Newton, so the result is valid for all
        // subsequent energy calls (line search) and gradient_hessian.
        if(m_has_inflate && !m_inflate_forces_computed)
        {
            compute_inflate_forces(info.x_prevs().viewer().name("x_prevs"));
            m_inflate_forces_computed = true;
        }

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(info.energies().size(),
                   [stencils    = stencils.viewer().name("stencils"),
                    drag_coeffs = drag_coefficients.viewer().name("drag_coefficients"),
                    has_curv    = m_has_curvature,
                    curv_scales = curvature_scales.viewer().name("curv_scales"),
                    H_v         = vertex_H_v.viewer().name("H_v"),
                    has_infl    = m_has_inflate,
                    infl_force  = tri_inflate_force.viewer().name("infl_force"),
                    xs          = info.xs().viewer().name("xs"),
                    x_prevs     = info.x_prevs().viewer().name("x_prevs"),
                    energies    = info.energies().viewer().name("energies"),
                    dt          = info.dt()] __device__(int I)
                   {
                       Vector3i stencil = stencils(I);
                       Float    coeff   = drag_coeffs(I);

                       if(has_curv)
                       {
                           Float cs = curv_scales(I);
                           if(cs != 0.0)
                           {
                               Float H_v_avg = (H_v(stencil[0]) + H_v(stencil[1])
                                                + H_v(stencil[2])) / 3.0;
                               coeff *= max(1.0 + cs * H_v_avg, (Float)0.0);
                           }
                       }

                       Vector3 x0 = xs(stencil[0]), x1 = xs(stencil[1]), x2 = xs(stencil[2]);
                       Vector3 x0p = x_prevs(stencil[0]), x1p = x_prevs(stencil[1]), x2p = x_prevs(stencil[2]);

                       Float E = AD::E(x0, x1, x2, x0p, x1p, x2p, coeff);

                       // Inflate: linear energy E_inflate = -F . x  for each vertex
                       if(has_infl)
                       {
                           Vector3 F = infl_force(I);
                           E += -F.dot(x0) - F.dot(x1) - F.dot(x2);
                       }

                       energies(I) = E;
                   });
    }

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        using namespace muda;
        namespace AD = sym::aerodynamic_damping;

        if(m_has_curvature)
        {
            LAUNCH_CURVATURE_DRAG_KERNEL(info.xs().viewer().name("xs"),
                                         info.x_prevs().viewer().name("x_prevs"));
        }

        // Reset inflate flag so forces are recomputed at the next Newton
        // iteration's first energy call (x_prevs doesn't actually change
        // within a timestep, but this keeps the invariant clean).
        m_inflate_forces_computed = false;

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(stencils.size(),
                   [stencils      = stencils.viewer().name("stencils"),
                    drag_coeffs   = drag_coefficients.viewer().name("drag_coefficients"),
                    has_curv      = m_has_curvature,
                    curv_scales   = curvature_scales.viewer().name("curv_scales"),
                    H_v           = vertex_H_v.viewer().name("H_v"),
                    has_infl      = m_has_inflate,
                    infl_force    = tri_inflate_force.viewer().name("infl_force"),
                    xs            = info.xs().viewer().name("xs"),
                    x_prevs       = info.x_prevs().viewer().name("x_prevs"),
                    G3s           = info.gradients().viewer().name("G3s"),
                    H3x3s        = info.hessians().viewer().name("H3x3s"),
                    gradient_only = info.gradient_only(),
                    dt            = info.dt()] __device__(int I)
                   {
                       Vector3i stencil = stencils(I);
                       Float    coeff   = drag_coeffs(I);

                       if(has_curv)
                       {
                           Float cs = curv_scales(I);
                           if(cs != 0.0)
                           {
                               Float H_v_avg = (H_v(stencil[0]) + H_v(stencil[1])
                                                + H_v(stencil[2])) / 3.0;
                               coeff *= max(1.0 + cs * H_v_avg, (Float)0.0);
                           }
                       }

                       Vector3 x0 = xs(stencil[0]), x1 = xs(stencil[1]), x2 = xs(stencil[2]);
                       Vector3 x0p = x_prevs(stencil[0]), x1p = x_prevs(stencil[1]), x2p = x_prevs(stencil[2]);

                       // Drag gradient
                       Vector<Float, 9> G;
                       AD::dEdx(G, x0, x1, x2, x0p, x1p, x2p, coeff);

                       // Inflate gradient: -F per vertex (constant)
                       if(has_infl)
                       {
                           Vector3 F = infl_force(I);
                           G.segment<3>(0) -= F;
                           G.segment<3>(3) -= F;
                           G.segment<3>(6) -= F;
                       }

                       DoubletVectorAssembler DVA{G3s};
                       DVA.segment<StencilSize>(I * StencilSize).write(stencil, G);

                       if(gradient_only)
                           return;

                       // Drag Hessian (inflate adds zero Hessian — linear energy)
                       Matrix<Float, 9, 9> H;
                       AD::ddEddx(H, x0, x1, x2, x0p, x1p, x2p, coeff);

                       TripletMatrixAssembler TMA{H3x3s};
                       TMA.half_block<StencilSize>(I * HalfHessianSize)
                           .write(stencil, H);
                   });
    }

#undef LAUNCH_CURVATURE_DRAG_KERNEL
};

REGISTER_SIM_SYSTEM(AerodynamicDamping);
}  // namespace uipc::backend::cuda
