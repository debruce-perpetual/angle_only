#pragma once

#include "angle_only/core/types.hpp"
#include <Eigen/Dense>

namespace aot {

/// Modified Spherical Coordinates (MSC) state vector:
///   [az, el, 1/r, vaz, vel, vr] — azimuth, elevation, inverse range,
///   and their rates. Used in angle-only tracking where range is unobservable
///   from a single sensor.
struct MSCState {
    Vec6 x = Vec6::Zero();      // state vector
    Mat6 P = Mat6::Identity();   // covariance

    // Accessors
    [[nodiscard]] double az() const { return x(0); }
    [[nodiscard]] double el() const { return x(1); }
    [[nodiscard]] double inv_range() const { return x(2); }
    [[nodiscard]] double az_rate() const { return x(3); }
    [[nodiscard]] double el_rate() const { return x(4); }
    [[nodiscard]] double inv_range_rate() const { return x(5); }

    void set_az(double v) { x(0) = v; }
    void set_el(double v) { x(1) = v; }
    void set_inv_range(double v) { x(2) = v; }
    void set_az_rate(double v) { x(3) = v; }
    void set_el_rate(double v) { x(4) = v; }
    void set_inv_range_rate(double v) { x(5) = v; }
};

/// Cartesian state vector: [x, y, z, vx, vy, vz]
struct CartesianState {
    Vec6 x = Vec6::Zero();
    Mat6 P = Mat6::Identity();

    [[nodiscard]] Vec3 position() const { return x.head<3>(); }
    [[nodiscard]] Vec3 velocity() const { return x.tail<3>(); }

    void set_position(const Vec3& pos) { x.head<3>() = pos; }
    void set_velocity(const Vec3& vel) { x.tail<3>() = vel; }
};

/// Convert MSC state to Cartesian position (relative to observer)
[[nodiscard]] Vec3 msc_to_cartesian(double az, double el, double inv_range);

/// Convert Cartesian position to MSC components
struct MSCComponents {
    double az;
    double el;
    double inv_range;
};
[[nodiscard]] MSCComponents cartesian_to_msc(const Vec3& pos);

} // namespace aot
