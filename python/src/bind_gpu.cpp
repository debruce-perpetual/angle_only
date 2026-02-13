#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "angle_only/gpu/dispatch.hpp"

namespace py = pybind11;

void bind_gpu(py::module_& m) {
    auto gpu = m.def_submodule("gpu", "GPU acceleration");

    gpu.def("cuda_available", &aot::gpu::cuda_available,
        "Check if CUDA is available at runtime");
    gpu.def("device_count", &aot::gpu::device_count,
        "Get number of CUDA devices");
    gpu.attr("gpu_batch_threshold") = aot::gpu::gpu_batch_threshold;
    gpu.def("should_use_gpu", &aot::gpu::should_use_gpu,
        py::arg("batch_size"),
        "Check if GPU dispatch is beneficial for given batch size");

    gpu.def("batch_predict", &aot::gpu::batch_predict,
        py::arg("states"), py::arg("covariances"),
        py::arg("F"), py::arg("Q"), py::arg("dt"));
    gpu.def("batch_correct", &aot::gpu::batch_correct,
        py::arg("states"), py::arg("covariances"),
        py::arg("measurements"), py::arg("H"), py::arg("R"));
    gpu.def("batch_likelihood", &aot::gpu::batch_likelihood,
        py::arg("states"), py::arg("covariances"),
        py::arg("measurements"), py::arg("H"), py::arg("R"),
        py::arg("likelihoods"));
}
