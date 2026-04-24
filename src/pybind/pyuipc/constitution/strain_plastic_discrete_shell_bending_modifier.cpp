#include <pyuipc/constitution/strain_plastic_discrete_shell_bending_modifier.h>
#include <uipc/constitution/constraint.h>
#include <uipc/constitution/strain_plastic_discrete_shell_bending_modifier.h>

namespace pyuipc::constitution
{
using namespace uipc::constitution;

PyStrainPlasticDiscreteShellBendingModifier::PyStrainPlasticDiscreteShellBendingModifier(
    py::module& m)
{
    auto class_ = py::class_<StrainPlasticDiscreteShellBendingModifier, Constraint>(
        m,
        "StrainPlasticDiscreteShellBendingModifier",
        R"(Zero-energy constraint that freezes plasticity in a StrainPlasticDiscreteShellBending mesh.

Set the per-mesh meta attribute "cancel_plastic" to 1 at any point
during simulation to freeze the plastic evolution (yield_threshold -> 1e30) and
optionally update the bending stiffness to "target_bending_stiffness".)");

    class_.def(
        py::init<const Json&>(),
        py::arg("config") = StrainPlasticDiscreteShellBendingModifier::default_config(),
        R"(Create a StrainPlasticDiscreteShellBendingModifier.
Args:
    config: Configuration dictionary (optional, uses default if not provided).)");

    class_.def_static(
        "default_config",
        &StrainPlasticDiscreteShellBendingModifier::default_config,
        R"(Get the default configuration.
Returns:
    dict: Default configuration dictionary.)");

    class_.def(
        "apply_to",
        &StrainPlasticDiscreteShellBendingModifier::apply_to,
        py::arg("sc"),
        R"(Register the modifier on a simplicial complex.
Args:
    sc: SimplicialComplex to apply to (must also have StrainPlasticDiscreteShellBending applied).
    Sets meta attributes "cancel_plastic" (default 0) and
    "target_bending_stiffness" (default 0.0) that can be written from animation
    callbacks to trigger freeze and change stiffness at runtime.)");
}
}  // namespace pyuipc::constitution
