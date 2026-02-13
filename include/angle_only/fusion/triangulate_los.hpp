#pragma once

#include "angle_only/core/types.hpp"
#include "angle_only/core/detection.hpp"
#include <vector>
#include <optional>

namespace aot::fusion {

/// Result of LOS triangulation
struct TriangulationResult {
    Vec3 position;          // estimated target position
    Mat3 covariance;        // position uncertainty
    double residual = 0.0;  // sum of squared residuals
    bool valid = false;
};

/// Triangulate target position from multiple line-of-sight measurements.
/// Uses linear least-squares (MATLAB's triangulateLOS).
/// Requires at least 2 LOS measurements from different positions.
[[nodiscard]] TriangulationResult triangulate_los(
    const std::vector<LOSMeasurement>& measurements);

/// Triangulate from sensor poses and detections
[[nodiscard]] TriangulationResult triangulate_los(
    const std::vector<SensorPose>& sensors,
    const std::vector<Detection>& detections);

} // namespace aot::fusion
