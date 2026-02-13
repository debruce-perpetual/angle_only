#pragma once

#include "angle_only/core/types.hpp"

namespace aot::motion {

/// Constant Velocity model in Cartesian coordinates.
/// State: [x, y, z, vx, vy, vz]
class ConstantVelocity {
public:
    static constexpr int state_dim = 6;

    explicit ConstantVelocity(double process_noise_intensity = 1.0);

    [[nodiscard]] Vec6 predict(const Vec6& state, double dt) const;
    [[nodiscard]] Mat6 jacobian(const Vec6& state, double dt) const;
    [[nodiscard]] Mat6 process_noise(double dt) const;

    void set_process_noise_intensity(double q) { q_ = q; }
    [[nodiscard]] double process_noise_intensity() const { return q_; }

private:
    double q_;
};

} // namespace aot::motion
