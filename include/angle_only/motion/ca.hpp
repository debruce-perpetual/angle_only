#pragma once

#include "angle_only/core/types.hpp"
#include <Eigen/Dense>

namespace aot::motion {

/// Constant Acceleration model in Cartesian coordinates.
/// State: [x, y, z, vx, vy, vz, ax, ay, az] — 9D state
class ConstantAcceleration {
public:
    static constexpr int state_dim = 9;

    using State = Eigen::Matrix<double, 9, 1>;
    using StateMat = Eigen::Matrix<double, 9, 9, Eigen::RowMajor>;

    explicit ConstantAcceleration(double process_noise_intensity = 1.0);

    [[nodiscard]] State predict(const State& state, double dt) const;
    [[nodiscard]] StateMat jacobian(const State& state, double dt) const;
    [[nodiscard]] StateMat process_noise(double dt) const;

    void set_process_noise_intensity(double q) { q_ = q; }
    [[nodiscard]] double process_noise_intensity() const { return q_; }

private:
    double q_;
};

} // namespace aot::motion
