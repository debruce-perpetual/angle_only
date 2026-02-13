#pragma once

#include "angle_only/core/types.hpp"
#include <vector>

namespace aot::association {

/// Mahalanobis distance between innovation y and covariance S
[[nodiscard]] double mahalanobis_distance(const VecX& y, const MatXR& S);

/// Ellipsoidal gating: returns indices of measurements passing the gate
[[nodiscard]] std::vector<int> gate(
    const VecX& predicted_measurement,
    const std::vector<VecX>& measurements,
    const MatXR& innovation_covariance,
    double threshold);

/// Chi-squared gate threshold for given dimension and confidence level
[[nodiscard]] double chi2_gate(int dimension, double confidence = 0.99);

} // namespace aot::association
