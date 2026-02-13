#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "angle_only/motion/constant_velocity_msc.hpp"
#include "angle_only/motion/cv.hpp"
#include "angle_only/motion/ca.hpp"
#include "angle_only/motion/ct.hpp"

namespace py = pybind11;

void bind_motion(py::module_& m) {
    auto motion = m.def_submodule("motion", "Motion models");

    py::class_<aot::motion::ConstantVelocityMSC>(motion, "ConstantVelocityMSC")
        .def(py::init<double>(), py::arg("process_noise_intensity") = 1.0)
        .def("predict", &aot::motion::ConstantVelocityMSC::predict,
            py::arg("state"), py::arg("dt"))
        .def("jacobian", &aot::motion::ConstantVelocityMSC::jacobian,
            py::arg("state"), py::arg("dt"))
        .def("process_noise", &aot::motion::ConstantVelocityMSC::process_noise,
            py::arg("dt"))
        .def_property("process_noise_intensity",
            &aot::motion::ConstantVelocityMSC::process_noise_intensity,
            &aot::motion::ConstantVelocityMSC::set_process_noise_intensity)
        .def_readonly_static("state_dim", &aot::motion::ConstantVelocityMSC::state_dim);

    py::class_<aot::motion::ConstantVelocity>(motion, "ConstantVelocity")
        .def(py::init<double>(), py::arg("process_noise_intensity") = 1.0)
        .def("predict", &aot::motion::ConstantVelocity::predict,
            py::arg("state"), py::arg("dt"))
        .def("jacobian", &aot::motion::ConstantVelocity::jacobian,
            py::arg("state"), py::arg("dt"))
        .def("process_noise", &aot::motion::ConstantVelocity::process_noise,
            py::arg("dt"))
        .def_property("process_noise_intensity",
            &aot::motion::ConstantVelocity::process_noise_intensity,
            &aot::motion::ConstantVelocity::set_process_noise_intensity);

    py::class_<aot::motion::ConstantAcceleration>(motion, "ConstantAcceleration")
        .def(py::init<double>(), py::arg("process_noise_intensity") = 1.0)
        .def("predict", &aot::motion::ConstantAcceleration::predict,
            py::arg("state"), py::arg("dt"))
        .def("jacobian", &aot::motion::ConstantAcceleration::jacobian,
            py::arg("state"), py::arg("dt"))
        .def("process_noise", &aot::motion::ConstantAcceleration::process_noise,
            py::arg("dt"));

    py::class_<aot::motion::CoordinatedTurn>(motion, "CoordinatedTurn")
        .def(py::init<double>(), py::arg("process_noise_intensity") = 1.0)
        .def("predict", &aot::motion::CoordinatedTurn::predict,
            py::arg("state"), py::arg("dt"))
        .def("jacobian", &aot::motion::CoordinatedTurn::jacobian,
            py::arg("state"), py::arg("dt"))
        .def("process_noise", &aot::motion::CoordinatedTurn::process_noise,
            py::arg("dt"));
}
