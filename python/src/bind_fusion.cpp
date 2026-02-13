#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "angle_only/fusion/triangulate_los.hpp"
#include "angle_only/fusion/static_detection_fuser.hpp"

namespace py = pybind11;

void bind_fusion(py::module_& m) {
    auto fusion = m.def_submodule("fusion", "Sensor fusion");

    // TriangulationResult
    py::class_<aot::fusion::TriangulationResult>(fusion, "TriangulationResult")
        .def_readwrite("position", &aot::fusion::TriangulationResult::position)
        .def_readwrite("covariance", &aot::fusion::TriangulationResult::covariance)
        .def_readwrite("residual", &aot::fusion::TriangulationResult::residual)
        .def_readwrite("valid", &aot::fusion::TriangulationResult::valid);

    // triangulate_los (LOS measurements overload)
    fusion.def("triangulate_los",
        py::overload_cast<const std::vector<aot::LOSMeasurement>&>(
            &aot::fusion::triangulate_los),
        py::arg("measurements"),
        "Triangulate target position from LOS measurements");

    // triangulate_los (sensors + detections overload)
    fusion.def("triangulate_los",
        py::overload_cast<const std::vector<aot::SensorPose>&,
                          const std::vector<aot::Detection>&>(
            &aot::fusion::triangulate_los),
        py::arg("sensors"), py::arg("detections"),
        "Triangulate from sensor poses and detections");

    // StaticDetectionFuser::Config
    py::class_<aot::fusion::StaticDetectionFuser::Config>(fusion, "StaticDetectionFuserConfig")
        .def(py::init<>())
        .def_readwrite("max_distance", &aot::fusion::StaticDetectionFuser::Config::max_distance)
        .def_readwrite("min_detections", &aot::fusion::StaticDetectionFuser::Config::min_detections)
        .def_readwrite("use_triangulation", &aot::fusion::StaticDetectionFuser::Config::use_triangulation);

    // StaticDetectionFuser::FusedDetection
    py::class_<aot::fusion::StaticDetectionFuser::FusedDetection>(fusion, "FusedDetection")
        .def_readwrite("position", &aot::fusion::StaticDetectionFuser::FusedDetection::position)
        .def_readwrite("covariance", &aot::fusion::StaticDetectionFuser::FusedDetection::covariance)
        .def_readwrite("time", &aot::fusion::StaticDetectionFuser::FusedDetection::time)
        .def_readwrite("sensor_ids", &aot::fusion::StaticDetectionFuser::FusedDetection::sensor_ids);

    // StaticDetectionFuser
    py::class_<aot::fusion::StaticDetectionFuser>(fusion, "StaticDetectionFuser")
        .def(py::init<>())
        .def(py::init<aot::fusion::StaticDetectionFuser::Config>(), py::arg("config"))
        .def("fuse", &aot::fusion::StaticDetectionFuser::fuse,
            py::arg("sensors"), py::arg("detections_per_sensor"))
        .def_property("config",
            &aot::fusion::StaticDetectionFuser::config,
            &aot::fusion::StaticDetectionFuser::set_config);
}
