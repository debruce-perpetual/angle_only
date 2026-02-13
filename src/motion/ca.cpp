#include "angle_only/motion/ca.hpp"

namespace aot::motion {

ConstantAcceleration::ConstantAcceleration(double process_noise_intensity)
    : q_(process_noise_intensity) {}

ConstantAcceleration::State ConstantAcceleration::predict(const State& state, double dt) const {
    State x_pred;
    double dt2 = dt * dt;
    // pos += vel*dt + 0.5*acc*dt^2
    x_pred.head<3>() = state.head<3>() + state.segment<3>(3) * dt + state.tail<3>() * (0.5 * dt2);
    // vel += acc*dt
    x_pred.segment<3>(3) = state.segment<3>(3) + state.tail<3>() * dt;
    // acc = const
    x_pred.tail<3>() = state.tail<3>();
    return x_pred;
}

ConstantAcceleration::StateMat ConstantAcceleration::jacobian(const State& /*state*/, double dt) const {
    double dt2 = dt * dt;
    StateMat F = StateMat::Identity();
    for (int i = 0; i < 3; ++i) {
        F(i, i + 3) = dt;
        F(i, i + 6) = 0.5 * dt2;
        F(i + 3, i + 6) = dt;
    }
    return F;
}

ConstantAcceleration::StateMat ConstantAcceleration::process_noise(double dt) const {
    // Jerk-driven process noise for constant acceleration model
    double dt2 = dt * dt;
    double dt3 = dt2 * dt;
    double dt4 = dt3 * dt;
    double dt5 = dt4 * dt;

    StateMat Q = StateMat::Zero();
    for (int i = 0; i < 3; ++i) {
        Q(i, i) = q_ * dt5 / 20.0;
        Q(i, i + 3) = q_ * dt4 / 8.0;
        Q(i, i + 6) = q_ * dt3 / 6.0;
        Q(i + 3, i) = q_ * dt4 / 8.0;
        Q(i + 3, i + 3) = q_ * dt3 / 3.0;
        Q(i + 3, i + 6) = q_ * dt2 / 2.0;
        Q(i + 6, i) = q_ * dt3 / 6.0;
        Q(i + 6, i + 3) = q_ * dt2 / 2.0;
        Q(i + 6, i + 6) = q_ * dt;
    }
    return Q;
}

} // namespace aot::motion
