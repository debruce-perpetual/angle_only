#pragma once

#include "angle_only/core/types.hpp"

namespace aot::measurement {

/// Spherical measurement model — measures azimuth and elevation from
/// Cartesian state. Used with Cartesian motion models.
/// State: [x, y, z, vx, vy, vz]
/// Measurement: [az, el]
class SphericalMeasurement {
public:
    static constexpr int meas_dim = 2;

    explicit SphericalMeasurement(const Mat2& noise = Mat2::Identity() * 1e-4);

    /// Predicted measurement: h(x) = [atan2(y,x), asin(z/r)]
    [[nodiscard]] Vec2 predict(const Vec6& state) const;

    /// Measurement Jacobian: dh/dx (2x6)
    [[nodiscard]] Mat2x6 jacobian(const Vec6& state) const;

    /// Measurement noise covariance
    [[nodiscard]] const Mat2& noise() const { return R_; }
    void set_noise(const Mat2& R) { R_ = R; }

private:
    Mat2 R_;
};

} // namespace aot::measurement
