#include "angle_only/association/gating.hpp"
#include <cmath>
#include <algorithm>

namespace aot::association {

double mahalanobis_distance(const VecX& y, const MatXR& S) {
    return std::sqrt(y.transpose() * S.inverse() * y);
}

std::vector<int> gate(
    const VecX& predicted_measurement,
    const std::vector<VecX>& measurements,
    const MatXR& innovation_covariance,
    double threshold)
{
    std::vector<int> gated;
    MatXR S_inv = innovation_covariance.inverse();

    for (int i = 0; i < static_cast<int>(measurements.size()); ++i) {
        VecX y = measurements[static_cast<size_t>(i)] - predicted_measurement;
        double d2 = y.transpose() * S_inv * y;
        if (d2 <= threshold * threshold) {
            gated.push_back(i);
        }
    }
    return gated;
}

double chi2_gate(int dimension, double confidence) {
    // Approximate chi-squared quantile using Wilson-Hilferty transform
    // For common cases, use lookup
    if (dimension == 2) {
        if (confidence >= 0.999) return 13.816;
        if (confidence >= 0.99) return 9.210;
        if (confidence >= 0.95) return 5.991;
        if (confidence >= 0.90) return 4.605;
    }
    if (dimension == 3) {
        if (confidence >= 0.999) return 16.266;
        if (confidence >= 0.99) return 11.345;
        if (confidence >= 0.95) return 7.815;
        if (confidence >= 0.90) return 6.251;
    }

    // Wilson-Hilferty approximation for other cases
    double p = confidence;
    double k = static_cast<double>(dimension);
    // Normal quantile approximation (Abramowitz & Stegun)
    double t = std::sqrt(-2.0 * std::log(1.0 - p));
    double z = t - (2.515517 + t * (0.802853 + t * 0.010328)) /
                    (1.0 + t * (1.432788 + t * (0.189269 + t * 0.001308)));
    double w = 1.0 - 2.0 / (9.0 * k) + z * std::sqrt(2.0 / (9.0 * k));
    return k * w * w * w;
}

} // namespace aot::association
