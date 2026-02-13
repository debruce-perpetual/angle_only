#pragma once

#include "angle_only/core/types.hpp"

namespace aot::coords {

/// Transform a position from sensor (body) frame to world (ENU) frame
[[nodiscard]] inline Vec3 sensor_to_world(
    const Vec3& pos_sensor,
    const Vec3& sensor_position,
    const Mat3& sensor_orientation)
{
    return sensor_position + sensor_orientation * pos_sensor;
}

/// Transform a position from world (ENU) frame to sensor (body) frame
[[nodiscard]] inline Vec3 world_to_sensor(
    const Vec3& pos_world,
    const Vec3& sensor_position,
    const Mat3& sensor_orientation)
{
    return sensor_orientation.transpose() * (pos_world - sensor_position);
}

/// Transform a direction vector from sensor to world frame
[[nodiscard]] inline Vec3 direction_sensor_to_world(
    const Vec3& dir_sensor,
    const Mat3& sensor_orientation)
{
    return sensor_orientation * dir_sensor;
}

/// Transform a direction vector from world to sensor frame
[[nodiscard]] inline Vec3 direction_world_to_sensor(
    const Vec3& dir_world,
    const Mat3& sensor_orientation)
{
    return sensor_orientation.transpose() * dir_world;
}

} // namespace aot::coords
