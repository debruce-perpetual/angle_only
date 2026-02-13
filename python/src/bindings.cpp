#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_core(py::module_& m);
void bind_coords(py::module_& m);
void bind_motion(py::module_& m);
void bind_measurement(py::module_& m);
void bind_filters(py::module_& m);
void bind_fusion(py::module_& m);
void bind_association(py::module_& m);
void bind_gpu(py::module_& m);

PYBIND11_MODULE(_angle_only_cpp, m) {
    m.doc() = "Angle-only tracking library — C++ bindings";

    bind_core(m);
    bind_coords(m);
    bind_motion(m);
    bind_measurement(m);
    bind_filters(m);
    bind_fusion(m);
    bind_association(m);
    bind_gpu(m);
}
