#include "angle_only/association/jpda.hpp"
#include <cmath>
#include <algorithm>

namespace aot::association {

JPDAResult jpda_probabilities(
    const MatXR& likelihood_matrix,
    double p_detection,
    double p_gate,
    double clutter_density)
{
    auto n_tracks = likelihood_matrix.rows();
    auto n_meas = likelihood_matrix.cols();

    JPDAResult result;
    result.beta = MatXR::Zero(n_tracks, n_meas + 1);

    for (Eigen::Index i = 0; i < n_tracks; ++i) {
        // For each track, compute marginal association probabilities
        double sum = 0.0;

        // Probability of no measurement (missed detection)
        double p_miss = (1.0 - p_detection * p_gate);

        for (Eigen::Index j = 0; j < n_meas; ++j) {
            double p_ij = p_detection * likelihood_matrix(i, j) /
                         (clutter_density + 1e-300);
            result.beta(i, j) = p_ij;
            sum += p_ij;
        }

        // Normalize
        sum += p_miss;
        if (sum > 1e-300) {
            for (Eigen::Index j = 0; j < n_meas; ++j) {
                result.beta(i, j) /= sum;
            }
            result.beta(i, n_meas) = p_miss / sum;
        } else {
            result.beta(i, n_meas) = 1.0;
        }
    }

    return result;
}

} // namespace aot::association
