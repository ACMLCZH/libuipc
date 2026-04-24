#pragma once
#include <uipc/constitution/constraint.h>
#include <uipc/geometry/simplicial_complex.h>
#include <uipc/common/json.h>

namespace uipc::constitution
{
/**
 * @brief A zero-energy constraint that freezes plasticity and optionally
 * raises the bending stiffness of a StrainPlasticDiscreteShellBending mesh.
 *
 * Apply this alongside StrainPlasticDiscreteShellBending. At any point during
 * the simulation set the per-mesh meta attribute "cancel_plastic" to 1
 * (e.g. via `view(sc.meta().find("cancel_plastic"))[0] = 1`).  The
 * backend will then set yield_threshold = 1e30 for every stencil belonging to
 * that mesh, permanently disabling further plastic evolution while keeping the
 * accumulated plastic rest-angle.  If "target_bending_stiffness" > 0 in the
 * meta, the stiffness is updated at the same time.
 */
class UIPC_CONSTITUTION_API StrainPlasticDiscreteShellBendingModifier final
    : public Constraint
{
    using Base = Constraint;

  public:
    StrainPlasticDiscreteShellBendingModifier(const Json& config = default_config()) noexcept;

    /**
     * @brief Register this modifier on a simplicial complex.
     *
     * @param sc                   The shell mesh already processed by
     *                             StrainPlasticDiscreteShellBending::apply_to.
     * @param new_bending_stiffness New bending stiffness to use after the
     *                             freeze is triggered.  Pass 0.0 (default) to
     *                             keep the original value.
     */
    void apply_to(geometry::SimplicialComplex& sc) const;

    static Json default_config();

  protected:
    U64 get_uid() const noexcept override;

  private:
    Json m_config;
};
}  // namespace uipc::constitution
