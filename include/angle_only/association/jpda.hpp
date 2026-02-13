#pragma once

#include "angle_only/core/types.hpp"
#include <vector>

namespace aot::association {

/// JPDA association probabilities.
/// Given a cost/likelihood matrix, compute the probability that each
/// measurement originated from each track (or clutter).
struct JPDAResult {
    /// beta(i,j) = probability that measurement j originated from track i
    /// beta(i, n_meas) = probability that track i has no measurement (missed detection)
    MatXR beta;
};

/// Compute JPDA association probabilities from a likelihood matrix.
/// likelihood(i,j) = likelihood of measurement j given track i.
/// p_detection = probability of detection, p_gate = probability of gated measurement.
[[nodiscard]] JPDAResult jpda_probabilities(
    const MatXR& likelihood_matrix,
    double p_detection = 0.9,
    double p_gate = 0.99,
    double clutter_density = 1e-6);

} // namespace aot::association
