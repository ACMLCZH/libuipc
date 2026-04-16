#include <pyuipc/constitution/aerodynamic_damping.h>
#include <uipc/constitution/finite_element_extra_constitution.h>
#include <uipc/constitution/aerodynamic_damping.h>

namespace pyuipc::constitution
{
using namespace uipc::constitution;

PyAerodynamicDamping::PyAerodynamicDamping(py::module& m)
{
    auto cls = py::class_<AerodynamicDamping, FiniteElementExtraConstitution>(
        m,
        "AerodynamicDamping",
        R"(Aerodynamic damping constitution for cloth air resistance.

Models drag force proportional to the face-normal velocity component
and face area. When cloth moves, faces perpendicular to the velocity
experience maximum drag (like opening a bag), while faces parallel
to the velocity experience minimal drag.)");

    cls.def(py::init<const Json&>(),
            py::arg("config") = AerodynamicDamping::default_config());

    cls.def_static("default_config", &AerodynamicDamping::default_config);

    cls.def("apply_to",
            &AerodynamicDamping::apply_to,
            py::arg("sc"),
            py::arg("drag_coefficient") = 0.5,
            py::arg("curvature_scale")  = 0.0,
            py::arg("inflate_scale")    = 0.0,
            R"(Apply aerodynamic damping to a cloth mesh.

Parameters
----------
sc : SimplicialComplex
    The triangle mesh to apply damping to.
drag_coefficient : float
    Combined drag coefficient (C_d * rho_air / 2). Default 0.5.
curvature_scale : float
    Scale factor for curvature-based drag modulation. 0 = no effect (default).
    Positive values amplify drag on concave regions (parachute) and reduce
    it on convex regions (dome).
inflate_scale : float
    Scale factor for curvature-based inflation pressure. 0 = no effect (default).
    When H_v > 0 (concave surface facing velocity), adds outward inflation force.)");
}
}  // namespace pyuipc::constitution
