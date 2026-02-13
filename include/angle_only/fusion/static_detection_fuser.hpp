#pragma once

#include "angle_only/core/types.hpp"
#include "angle_only/core/detection.hpp"
#include "angle_only/fusion/triangulate_los.hpp"
#include <vector>
#include <functional>

namespace aot::fusion {

/// Fuses detections from multiple sensors into position estimates.
/// Implements a simplified version of MATLAB's staticDetectionFuser.
class StaticDetectionFuser {
public:
    struct Config {
        double max_distance = 0.1;
        int min_detections = 2;
        bool use_triangulation = true;
        Config() = default;
    };

    StaticDetectionFuser();
    explicit StaticDetectionFuser(Config config);

    struct FusedDetection {
        Vec3 position;
        Mat3 covariance;
        double time = 0.0;
        std::vector<uint32_t> sensor_ids;
    };

    /// Fuse detections from multiple sensors at a given time
    [[nodiscard]] std::vector<FusedDetection> fuse(
        const std::vector<SensorPose>& sensors,
        const std::vector<std::vector<Detection>>& detections_per_sensor) const;

    [[nodiscard]] const Config& config() const { return config_; }
    void set_config(const Config& c) { config_ = c; }

private:
    Config config_;
};

} // namespace aot::fusion
