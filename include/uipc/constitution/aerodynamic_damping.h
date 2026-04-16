#pragma once
#include <uipc/constitution/finite_element_extra_constitution.h>
#include <uipc/common/unit.h>

namespace uipc::constitution
{
class UIPC_CONSTITUTION_API AerodynamicDamping : public FiniteElementExtraConstitution
{
    using Base = FiniteElementExtraConstitution;

  public:
    AerodynamicDamping(const Json& json = default_config());

    /**
     * @brief Apply aerodynamic damping to a cloth mesh.
     *
     * Models air resistance as a dissipative force proportional to
     * the face-normal velocity component and face area.
     *
     * When curvature_scale > 0, the drag coefficient is modulated by the
     * local mean curvature projected onto the velocity direction:
     *   effective_coeff = coeff * max(1 + curvature_scale * H_v, 0)
     * This makes concave surfaces (e.g. parachutes) experience more drag
     * than convex surfaces (e.g. domes), matching real-world aerodynamics.
     *
     * @param sc The simplicial complex (triangle mesh).
     * @param drag_coefficient Combined aerodynamic drag coefficient (C_d * rho_air / 2).
     *        Default 0.5 (moderate drag for cloth-like materials).
     * @param curvature_scale Scale factor for curvature-based drag modulation.
     *        0 = no curvature effect (default). Positive values amplify drag
     *        on concave regions and reduce it on convex regions.
     * @param inflate_scale Scale factor for curvature-based inflation pressure.
     *        When H_v > 0 (concave surface facing velocity, e.g. parachute interior),
     *        adds outward inflation force: E_inflate = -K * alpha / ||c||
     *        where K = inflate_scale * avg(max(H_v, 0)).
     *        0 = no inflation (default).
     */
    void apply_to(geometry::SimplicialComplex& sc,
                  Float                        drag_coefficient = 0.5,
                  Float                        curvature_scale  = 0.0,
                  Float                        inflate_scale    = 0.0);

    static Json default_config();

  private:
    virtual U64 get_uid() const noexcept final override;
    Json        m_config;
};
}  // namespace uipc::constitution
