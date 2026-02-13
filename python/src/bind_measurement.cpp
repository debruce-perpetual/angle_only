#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "angle_only/measurement/msc_measurement.hpp"
#include "angle_only/measurement/spherical_measurement.hpp"

namespace py = pybind11;

void bind_measurement(py::module_& m) {
    auto meas = m.def_submodule("measurement", "Measurement models");

    py::class_<aot::measurement::MSCMeasurement>(meas, "MSCMeasurement")
        .def(py::init<>())
        .def(py::init<const aot::Mat2&>(), py::arg("noise"))
        .def("predict", &aot::measurement::MSCMeasurement::predict, py::arg("state"))
        .def("jacobian", &aot::measurement::MSCMeasurement::jacobian, py::arg("state"))
        .def_property("noise",
            &aot::measurement::MSCMeasurement::noise,
            &aot::measurement::MSCMeasurement::set_noise);

    py::class_<aot::measurement::SphericalMeasurement>(meas, "SphericalMeasurement")
        .def(py::init<>())
        .def(py::init<const aot::Mat2&>(), py::arg("noise"))
        .def("predict", &aot::measurement::SphericalMeasurement::predict, py::arg("state"))
        .def("jacobian", &aot::measurement::SphericalMeasurement::jacobian, py::arg("state"))
        .def_property("noise",
            &aot::measurement::SphericalMeasurement::noise,
            &aot::measurement::SphericalMeasurement::set_noise);
}
