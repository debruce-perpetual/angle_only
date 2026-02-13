#pragma once

#include "angle_only/core/types.hpp"

namespace aot::measurement {

/// MSC measurement model — measures azimuth and elevation from state.
/// Implements MATLAB's cvmeasmsc / cvmeasmscjac.
/// State: [az, el, 1/r, az_rate, el_rate, inv_range_rate]
/// Measurement: [az, el]
class MSCMeasurement {
public:
    static constexpr int meas_dim = 2;

    explicit MSCMeasurement(const Mat2& noise = Mat2::Identity() * 1e-4);

    /// Predicted measurement from state: h(x) = [az, el]
    [[nodiscard]] Vec2 predict(const Vec6& state) const;

    /// Measurement Jacobian: dh/dx
    [[nodiscard]] Mat2x6 jacobian(const Vec6& state) const;

    /// Measurement noise covariance
    [[nodiscard]] const Mat2& noise() const { return R_; }
    void set_noise(const Mat2& R) { R_ = R; }

private:
    Mat2 R_;
};

} // namespace aot::measurement
