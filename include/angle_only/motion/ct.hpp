#pragma once

#include "angle_only/core/types.hpp"
#include <Eigen/Dense>

namespace aot::motion {

/// Coordinated Turn model in Cartesian coordinates.
/// State: [x, y, z, vx, vy, vz, omega] — 7D state, omega = turn rate
class CoordinatedTurn {
public:
    static constexpr int state_dim = 7;

    using State = Eigen::Matrix<double, 7, 1>;
    using StateMat = Eigen::Matrix<double, 7, 7, Eigen::RowMajor>;

    explicit CoordinatedTurn(double process_noise_intensity = 1.0);

    [[nodiscard]] State predict(const State& state, double dt) const;
    [[nodiscard]] StateMat jacobian(const State& state, double dt) const;
    [[nodiscard]] StateMat process_noise(double dt) const;

    void set_process_noise_intensity(double q) { q_ = q; }
    [[nodiscard]] double process_noise_intensity() const { return q_; }

private:
    double q_;
};

} // namespace aot::motion
