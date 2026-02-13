#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "angle_only/core/state.hpp"
#include "angle_only/core/detection.hpp"
#include "angle_only/core/constants.hpp"

namespace py = pybind11;

void bind_core(py::module_& m) {
    auto core = m.def_submodule("core", "Core types and constants");

    // Constants
    core.attr("pi") = aot::pi;
    core.attr("two_pi") = aot::two_pi;
    core.attr("deg2rad") = aot::deg2rad;
    core.attr("rad2deg") = aot::rad2deg;

    // MSCState
    py::class_<aot::MSCState>(core, "MSCState")
        .def(py::init<>())
        .def_readwrite("x", &aot::MSCState::x)
        .def_readwrite("P", &aot::MSCState::P)
        .def_property("az", &aot::MSCState::az, &aot::MSCState::set_az)
        .def_property("el", &aot::MSCState::el, &aot::MSCState::set_el)
        .def_property("inv_range", &aot::MSCState::inv_range, &aot::MSCState::set_inv_range)
        .def_property("az_rate", &aot::MSCState::az_rate, &aot::MSCState::set_az_rate)
        .def_property("el_rate", &aot::MSCState::el_rate, &aot::MSCState::set_el_rate)
        .def_property("inv_range_rate", &aot::MSCState::inv_range_rate, &aot::MSCState::set_inv_range_rate)
        .def("__repr__", [](const aot::MSCState& s) {
            return "<MSCState az=" + std::to_string(s.az()) +
                   " el=" + std::to_string(s.el()) +
                   " inv_r=" + std::to_string(s.inv_range()) + ">";
        });

    // CartesianState
    py::class_<aot::CartesianState>(core, "CartesianState")
        .def(py::init<>())
        .def_readwrite("x", &aot::CartesianState::x)
        .def_readwrite("P", &aot::CartesianState::P)
        .def_property("position",
            [](const aot::CartesianState& s) { return s.position(); },
            [](aot::CartesianState& s, const aot::Vec3& p) { s.set_position(p); })
        .def_property("velocity",
            [](const aot::CartesianState& s) { return s.velocity(); },
            [](aot::CartesianState& s, const aot::Vec3& v) { s.set_velocity(v); });

    // Detection
    py::class_<aot::Detection>(core, "Detection")
        .def(py::init<>())
        .def_readwrite("azimuth", &aot::Detection::azimuth)
        .def_readwrite("elevation", &aot::Detection::elevation)
        .def_readwrite("noise", &aot::Detection::noise)
        .def_readwrite("time", &aot::Detection::time)
        .def_readwrite("sensor_id", &aot::Detection::sensor_id);

    // SensorPose
    py::class_<aot::SensorPose>(core, "SensorPose")
        .def(py::init<>())
        .def_readwrite("position", &aot::SensorPose::position)
        .def_readwrite("velocity", &aot::SensorPose::velocity)
        .def_readwrite("orientation", &aot::SensorPose::orientation)
        .def_readwrite("sensor_id", &aot::SensorPose::sensor_id);

    // LOSMeasurement
    py::class_<aot::LOSMeasurement>(core, "LOSMeasurement")
        .def(py::init<>())
        .def_readwrite("origin", &aot::LOSMeasurement::origin)
        .def_readwrite("direction", &aot::LOSMeasurement::direction)
        .def_readwrite("noise", &aot::LOSMeasurement::noise)
        .def_readwrite("time", &aot::LOSMeasurement::time)
        .def_readwrite("sensor_id", &aot::LOSMeasurement::sensor_id);

    // Free functions
    core.def("msc_to_cartesian", &aot::msc_to_cartesian,
        py::arg("az"), py::arg("el"), py::arg("inv_range"),
        "Convert MSC coordinates to Cartesian position");
    core.def("cartesian_to_msc", &aot::cartesian_to_msc,
        py::arg("pos"), "Convert Cartesian position to MSC components");
}
