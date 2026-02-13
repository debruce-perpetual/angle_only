#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "angle_only/filters/msc_ekf.hpp"
#include "angle_only/filters/gmphd.hpp"

namespace py = pybind11;

void bind_filters(py::module_& m) {
    auto filters = m.def_submodule("filters", "Tracking filters");

    // MSCEKF
    py::class_<aot::filters::MSCEKF>(filters, "MSCEKF")
        .def(py::init<>())
        .def(py::init<const aot::MSCState&>(), py::arg("initial_state"))
        .def("predict", &aot::filters::MSCEKF::predict, py::arg("dt"))
        .def("correct", &aot::filters::MSCEKF::correct,
            py::arg("measurement"), py::arg("R"))
        .def("correct_jpda", &aot::filters::MSCEKF::correct_jpda,
            py::arg("measurements"), py::arg("weights"), py::arg("R"))
        .def("distance", &aot::filters::MSCEKF::distance,
            py::arg("z"), py::arg("R"))
        .def("likelihood", &aot::filters::MSCEKF::likelihood,
            py::arg("z"), py::arg("R"))
        .def("smooth", &aot::filters::MSCEKF::smooth)
        .def_property("state",
            [](const aot::filters::MSCEKF& f) { return f.state(); },
            &aot::filters::MSCEKF::set_state)
        .def_property("covariance",
            [](const aot::filters::MSCEKF& f) { return f.covariance(); },
            &aot::filters::MSCEKF::set_covariance)
        .def_property_readonly("msc_state", &aot::filters::MSCEKF::msc_state)
        .def("set_store_history", &aot::filters::MSCEKF::set_store_history,
            py::arg("store"))
        .def("clear_history", &aot::filters::MSCEKF::clear_history)
        .def_static("from_detection", [](const aot::Detection& det,
                                          double inv_range, double inv_range_std) {
            return aot::filters::initcvmscekf(det, inv_range, inv_range_std);
        }, py::arg("detection"),
           py::arg("initial_inv_range") = 0.01,
           py::arg("inv_range_std") = 0.05);

    filters.def("initcvmscekf", &aot::filters::initcvmscekf,
        py::arg("detection"),
        py::arg("initial_inv_range") = 0.01,
        py::arg("inv_range_std") = 0.05);

    // GaussianComponent
    py::class_<aot::filters::GaussianComponent>(filters, "GaussianComponent")
        .def(py::init<>())
        .def(py::init<double, aot::VecX, aot::MatXR>(),
            py::arg("weight"), py::arg("mean"), py::arg("covariance"))
        .def_readwrite("weight", &aot::filters::GaussianComponent::weight)
        .def_readwrite("mean", &aot::filters::GaussianComponent::mean)
        .def_readwrite("covariance", &aot::filters::GaussianComponent::covariance);

    // GMPHD Config
    py::class_<aot::filters::GMPHD::Config>(filters, "GMPHDConfig")
        .def(py::init<>())
        .def_readwrite("p_survival", &aot::filters::GMPHD::Config::p_survival)
        .def_readwrite("p_detection", &aot::filters::GMPHD::Config::p_detection)
        .def_readwrite("clutter_rate", &aot::filters::GMPHD::Config::clutter_rate)
        .def_readwrite("merge_threshold", &aot::filters::GMPHD::Config::merge_threshold)
        .def_readwrite("prune_threshold", &aot::filters::GMPHD::Config::prune_threshold)
        .def_readwrite("max_components", &aot::filters::GMPHD::Config::max_components)
        .def_readwrite("extraction_threshold", &aot::filters::GMPHD::Config::extraction_threshold);

    // GMPHD
    py::class_<aot::filters::GMPHD>(filters, "GMPHD")
        .def(py::init<>())
        .def(py::init<aot::filters::GMPHD::Config>(), py::arg("config"))
        .def("add_birth", &aot::filters::GMPHD::add_birth, py::arg("births"))
        .def("add_birth_from_detection", &aot::filters::GMPHD::add_birth_from_detection,
            py::arg("measurement"), py::arg("R_meas"), py::arg("range_hypotheses"))
        .def("predict", &aot::filters::GMPHD::predict, py::arg("dt"))
        .def("correct", &aot::filters::GMPHD::correct, py::arg("measurements"))
        .def("merge", &aot::filters::GMPHD::merge)
        .def("prune", &aot::filters::GMPHD::prune)
        .def("extract", &aot::filters::GMPHD::extract)
        .def_property_readonly("components", &aot::filters::GMPHD::components)
        .def_property_readonly("estimated_target_count", &aot::filters::GMPHD::estimated_target_count)
        .def_property("config",
            &aot::filters::GMPHD::config,
            &aot::filters::GMPHD::set_config);
}
