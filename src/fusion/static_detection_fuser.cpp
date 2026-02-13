#include "angle_only/fusion/static_detection_fuser.hpp"
#include "angle_only/coords/transforms.hpp"
#include "angle_only/coords/angle_wrap.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace aot::fusion {

StaticDetectionFuser::StaticDetectionFuser() : config_() {}
StaticDetectionFuser::StaticDetectionFuser(Config config) : config_(std::move(config)) {}

std::vector<StaticDetectionFuser::FusedDetection> StaticDetectionFuser::fuse(
    const std::vector<SensorPose>& sensors,
    const std::vector<std::vector<Detection>>& detections_per_sensor) const
{
    // Build all LOS measurements
    struct IndexedLOS {
        LOSMeasurement los;
        size_t sensor_idx;
        size_t det_idx;
        bool used = false;
    };

    std::vector<IndexedLOS> all_los;
    for (size_t s = 0; s < detections_per_sensor.size() && s < sensors.size(); ++s) {
        for (size_t d = 0; d < detections_per_sensor[s].size(); ++d) {
            const auto& det = detections_per_sensor[s][d];
            LOSMeasurement los;
            los.origin = sensors[s].position;
            los.direction = sensors[s].orientation *
                coords::az_el_to_unit_vector(det.azimuth, det.elevation);
            los.noise = det.noise;
            los.time = det.time;
            los.sensor_id = sensors[s].sensor_id;
            all_los.push_back({los, s, d, false});
        }
    }

    // Greedy association: group LOS measurements with similar directions
    std::vector<FusedDetection> results;

    for (size_t i = 0; i < all_los.size(); ++i) {
        if (all_los[i].used) continue;

        std::vector<LOSMeasurement> group;
        std::vector<uint32_t> group_sensors;
        group.push_back(all_los[i].los);
        group_sensors.push_back(all_los[i].los.sensor_id);
        all_los[i].used = true;

        for (size_t j = i + 1; j < all_los.size(); ++j) {
            if (all_los[j].used) continue;
            if (all_los[j].sensor_idx == all_los[i].sensor_idx) continue;

            // Check angular distance between directions
            double dot = all_los[i].los.direction.dot(all_los[j].los.direction);
            double angle = std::acos(std::clamp(dot, -1.0, 1.0));

            if (angle < config_.max_distance) {
                group.push_back(all_los[j].los);
                group_sensors.push_back(all_los[j].los.sensor_id);
                all_los[j].used = true;
            }
        }

        if (static_cast<int>(group.size()) >= config_.min_detections && config_.use_triangulation) {
            auto tri = triangulate_los(group);
            if (tri.valid) {
                FusedDetection fd;
                fd.position = tri.position;
                fd.covariance = tri.covariance;
                fd.time = group[0].time;
                fd.sensor_ids = std::move(group_sensors);
                results.push_back(std::move(fd));
            }
        }
    }

    return results;
}

} // namespace aot::fusion
