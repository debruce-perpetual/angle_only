#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "angle_only/coords/transforms.hpp"
#include "angle_only/coords/angle_wrap.hpp"
#include "angle_only/coords/frames.hpp"

namespace py = pybind11;

void bind_coords(py::module_& m) {
    auto coords = m.def_submodule("coords", "Coordinate transforms");

    coords.def("spherical_to_cartesian", &aot::coords::spherical_to_cartesian,
        py::arg("az"), py::arg("el"), py::arg("r"));
    coords.def("cartesian_to_spherical", [](const aot::Vec3& pos) {
        auto s = aot::coords::cartesian_to_spherical(pos);
        return py::make_tuple(s.az, s.el, s.r);
    }, py::arg("pos"));

    coords.def("msc_to_cartesian", &aot::coords::msc_to_cartesian,
        py::arg("az"), py::arg("el"), py::arg("inv_range"));
    coords.def("cartesian_to_msc", [](const aot::Vec3& pos) {
        auto m = aot::coords::cartesian_to_msc(pos);
        return py::make_tuple(m.az, m.el, m.inv_range);
    }, py::arg("pos"));

    coords.def("msc_to_cartesian_jacobian", &aot::coords::msc_to_cartesian_jacobian,
        py::arg("az"), py::arg("el"), py::arg("inv_range"));
    coords.def("cartesian_to_msc_jacobian", &aot::coords::cartesian_to_msc_jacobian,
        py::arg("pos"));

    coords.def("az_el_to_unit_vector", &aot::coords::az_el_to_unit_vector,
        py::arg("az"), py::arg("el"));
    coords.def("unit_vector_to_az_el", [](const aot::Vec3& dir) {
        auto ae = aot::coords::unit_vector_to_az_el(dir);
        return py::make_tuple(ae.az, ae.el);
    }, py::arg("dir"));

    coords.def("wrap_to_pi", &aot::coords::wrap_to_pi, py::arg("angle"));
    coords.def("wrap_to_2pi", &aot::coords::wrap_to_2pi, py::arg("angle"));
    coords.def("angle_diff", &aot::coords::angle_diff, py::arg("a"), py::arg("b"));

    coords.def("sensor_to_world", &aot::coords::sensor_to_world,
        py::arg("pos_sensor"), py::arg("sensor_position"), py::arg("sensor_orientation"));
    coords.def("world_to_sensor", &aot::coords::world_to_sensor,
        py::arg("pos_world"), py::arg("sensor_position"), py::arg("sensor_orientation"));

    coords.def("enu_to_body", &aot::coords::enu_to_body,
        py::arg("yaw"), py::arg("pitch"), py::arg("roll"));
    coords.def("body_to_enu", &aot::coords::body_to_enu,
        py::arg("yaw"), py::arg("pitch"), py::arg("roll"));
}
