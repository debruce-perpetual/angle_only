#pragma once

#include "angle_only/core/types.hpp"
#include <cstdint>

namespace aot {

/// An angular detection from a sensor
struct Detection {
    double azimuth = 0.0;       // radians
    double elevation = 0.0;     // radians
    Mat2 noise = Mat2::Identity() * 1e-4;  // measurement noise covariance
    double time = 0.0;          // timestamp
    uint32_t sensor_id = 0;     // originating sensor
};

/// Sensor platform state (position and optionally velocity)
struct SensorPose {
    Vec3 position = Vec3::Zero();
    Vec3 velocity = Vec3::Zero();
    Mat3 orientation = Mat3::Identity();  // body-to-world rotation
    uint32_t sensor_id = 0;
};

/// Line-of-sight measurement for triangulation
struct LOSMeasurement {
    Vec3 origin;      // sensor position in world frame
    Vec3 direction;   // unit direction vector
    Mat2 noise;       // angular noise covariance
    double time = 0.0;
    uint32_t sensor_id = 0;
};

} // namespace aot
