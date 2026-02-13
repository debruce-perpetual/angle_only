#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "angle_only/association/gating.hpp"
#include "angle_only/association/gnn.hpp"
#include "angle_only/association/jpda.hpp"

namespace py = pybind11;

void bind_association(py::module_& m) {
    auto assoc = m.def_submodule("association", "Data association");

    assoc.def("mahalanobis_distance", &aot::association::mahalanobis_distance,
        py::arg("y"), py::arg("S"));
    assoc.def("gate", &aot::association::gate,
        py::arg("predicted_measurement"), py::arg("measurements"),
        py::arg("innovation_covariance"), py::arg("threshold"));
    assoc.def("chi2_gate", &aot::association::chi2_gate,
        py::arg("dimension"), py::arg("confidence") = 0.99);

    // AssignmentResult
    py::class_<aot::association::AssignmentResult>(assoc, "AssignmentResult")
        .def_readwrite("assignments", &aot::association::AssignmentResult::assignments)
        .def_readwrite("unassigned_tracks", &aot::association::AssignmentResult::unassigned_tracks)
        .def_readwrite("unassigned_measurements", &aot::association::AssignmentResult::unassigned_measurements)
        .def_readwrite("total_cost", &aot::association::AssignmentResult::total_cost);

    assoc.def("gnn_assign", &aot::association::gnn_assign,
        py::arg("cost_matrix"), py::arg("gate_threshold") = 1e10);
    assoc.def("auction_assign", &aot::association::auction_assign,
        py::arg("cost_matrix"), py::arg("epsilon") = 1e-6,
        py::arg("gate_threshold") = 1e10);

    // JPDAResult
    py::class_<aot::association::JPDAResult>(assoc, "JPDAResult")
        .def_readwrite("beta", &aot::association::JPDAResult::beta);

    assoc.def("jpda_probabilities", &aot::association::jpda_probabilities,
        py::arg("likelihood_matrix"),
        py::arg("p_detection") = 0.9,
        py::arg("p_gate") = 0.99,
        py::arg("clutter_density") = 1e-6);
}
