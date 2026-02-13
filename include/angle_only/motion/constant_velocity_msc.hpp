#pragma once

#include "angle_only/core/types.hpp"

namespace aot::motion {

/// Constant velocity motion model in Modified Spherical Coordinates (MSC).
/// Implements the MATLAB constvelmsc / constvelmscjac functions.
/// State: [az, el, 1/r, az_rate, el_rate, inv_range_rate]
class ConstantVelocityMSC {
public:
    static constexpr int state_dim = 6;

    explicit ConstantVelocityMSC(double process_noise_intensity = 1.0);

    /// State transition: x(k+1) = f(x(k), dt)
    [[nodiscard]] Vec6 predict(const Vec6& state, double dt) const;

    /// Jacobian of state transition: df/dx
    [[nodiscard]] Mat6 jacobian(const Vec6& state, double dt) const;

    /// Process noise covariance Q(dt)
    [[nodiscard]] Mat6 process_noise(double dt) const;

    void set_process_noise_intensity(double q) { q_ = q; }
    [[nodiscard]] double process_noise_intensity() const { return q_; }

private:
    double q_;
};

} // namespace aot::motion
