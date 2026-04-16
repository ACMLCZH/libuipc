#include <uipc/constitution/aerodynamic_damping.h>
#include <uipc/builtin/constitution_type.h>
#include <uipc/builtin/constitution_uid_auto_register.h>

namespace uipc::constitution
{
constexpr U64 AerodynamicDampingUID = 33;

REGISTER_CONSTITUTION_UIDS()
{
    list<builtin::UIDInfo> uid_infos;
    builtin::UIDInfo       info;
    info.uid  = AerodynamicDampingUID;
    info.name = "AerodynamicDamping";
    info.type = string{builtin::FiniteElement};
    uid_infos.push_back(info);
    return uid_infos;
}

AerodynamicDamping::AerodynamicDamping(const Json& json)
    : m_config{json}
{
}

void AerodynamicDamping::apply_to(geometry::SimplicialComplex& sc,
                                   Float drag_coefficient_v,
                                   Float curvature_scale_v,
                                   Float inflate_scale_v)
{
    Base::apply_to(sc);

    auto dc = sc.triangles().find<Float>("drag_coefficient");
    if(!dc)
    {
        dc = sc.triangles().create<Float>("drag_coefficient");
    }
    auto dc_view = geometry::view(*dc);
    std::ranges::fill(dc_view, drag_coefficient_v);

    auto cs = sc.triangles().find<Float>("curvature_scale");
    if(!cs)
    {
        cs = sc.triangles().create<Float>("curvature_scale");
    }
    auto cs_view = geometry::view(*cs);
    std::ranges::fill(cs_view, curvature_scale_v);

    if(inflate_scale_v != 0.0)
    {
        auto is = sc.triangles().find<Float>("inflate_scale");
        if(!is)
        {
            is = sc.triangles().create<Float>("inflate_scale");
        }
        auto is_view = geometry::view(*is);
        std::ranges::fill(is_view, inflate_scale_v);
    }
}

U64 AerodynamicDamping::get_uid() const noexcept
{
    return AerodynamicDampingUID;
}

Json AerodynamicDamping::default_config()
{
    return Json::object();
}
}  // namespace uipc::constitution
