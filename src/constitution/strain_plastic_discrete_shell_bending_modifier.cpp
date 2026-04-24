#include <uipc/constitution/strain_plastic_discrete_shell_bending_modifier.h>
#include <uipc/builtin/constitution_uid_auto_register.h>
#include <uipc/builtin/constitution_type.h>
#include <uipc/geometry/simplicial_complex.h>

namespace uipc::constitution
{
constexpr U64 StrainPlasticDiscreteShellBendingModifierUID = 34;

REGISTER_CONSTITUTION_UIDS()
{
    using namespace builtin;
    list<UIDInfo> uids;
    uids.push_back(UIDInfo{.uid  = StrainPlasticDiscreteShellBendingModifierUID,
                           .name = "StrainPlasticDiscreteShellBendingModifier",
                           .type = string{builtin::Constraint}});
    return uids;
};

StrainPlasticDiscreteShellBendingModifier::StrainPlasticDiscreteShellBendingModifier(
    const Json& config) noexcept
    : m_config(config)
{
}

void StrainPlasticDiscreteShellBendingModifier::apply_to(geometry::SimplicialComplex& sc) const
{
    Base::apply_to(sc);

    // Per-edge: "cancel_plastic" — set to 1 on any edge to freeze that stencil.
    auto freeze_enabled = sc.edges().find<IndexT>("cancel_plastic");
    if(!freeze_enabled)
        freeze_enabled = sc.edges().create<IndexT>("cancel_plastic");
    std::ranges::fill(geometry::view(*freeze_enabled), IndexT(0));

    // Per-edge: "target_bending_stiffness" — new kappa after freeze (0 = keep original).
    auto target_stiffness = sc.edges().find<Float>("target_bending_stiffness");
    if(!target_stiffness)
        target_stiffness = sc.edges().create<Float>("target_bending_stiffness");
    std::ranges::fill(geometry::view(*target_stiffness), Float(0));
}

Json StrainPlasticDiscreteShellBendingModifier::default_config()
{
    return Json::object();
}

U64 StrainPlasticDiscreteShellBendingModifier::get_uid() const noexcept
{
    return StrainPlasticDiscreteShellBendingModifierUID;
}
}  // namespace uipc::constitution
