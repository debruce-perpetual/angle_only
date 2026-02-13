#pragma once

#include "angle_only/core/types.hpp"
#include <vector>
#include <utility>

namespace aot::association {

/// Assignment result: pairs of (track_index, measurement_index)
/// Unassigned tracks/measurements have index -1
struct AssignmentResult {
    std::vector<std::pair<int, int>> assignments;
    std::vector<int> unassigned_tracks;
    std::vector<int> unassigned_measurements;
    double total_cost = 0.0;
};

/// Global Nearest Neighbor (GNN) assignment using the Hungarian algorithm.
/// cost_matrix(i,j) = cost of assigning track i to measurement j.
/// gate_threshold: assignments with cost > threshold are forbidden.
[[nodiscard]] AssignmentResult gnn_assign(
    const MatXR& cost_matrix,
    double gate_threshold = 1e10);

/// Auction algorithm for assignment (faster for sparse problems)
[[nodiscard]] AssignmentResult auction_assign(
    const MatXR& cost_matrix,
    double epsilon = 1e-6,
    double gate_threshold = 1e10);

} // namespace aot::association
