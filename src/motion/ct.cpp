#include "angle_only/motion/ct.hpp"
#include <cmath>

namespace aot::motion {

CoordinatedTurn::CoordinatedTurn(double process_noise_intensity)
    : q_(process_noise_intensity) {}

CoordinatedTurn::State CoordinatedTurn::predict(const State& state, double dt) const {
    double vx = state(3), vy = state(4), vz = state(5);
    double omega = state(6);

    State x_pred;

    if (std::abs(omega) < 1e-10) {
        // Nearly zero turn rate — use linear model
        x_pred(0) = state(0) + vx * dt;
        x_pred(1) = state(1) + vy * dt;
        x_pred(2) = state(2) + vz * dt;
        x_pred(3) = vx;
        x_pred(4) = vy;
        x_pred(5) = vz;
        x_pred(6) = omega;
    } else {
        double s = std::sin(omega * dt);
        double c = std::cos(omega * dt);
        double so = s / omega;
        double co = (1.0 - c) / omega;

        x_pred(0) = state(0) + so * vx - co * vy;
        x_pred(1) = state(1) + co * vx + so * vy;
        x_pred(2) = state(2) + vz * dt;
        x_pred(3) = c * vx - s * vy;
        x_pred(4) = s * vx + c * vy;
        x_pred(5) = vz;
        x_pred(6) = omega;
    }
    return x_pred;
}

CoordinatedTurn::StateMat CoordinatedTurn::jacobian(const State& state, double dt) const {
    double vx = state(3), vy = state(4);
    double omega = state(6);

    StateMat F = StateMat::Zero();
    F(2, 2) = 1.0;
    F(2, 5) = dt;
    F(5, 5) = 1.0;
    F(6, 6) = 1.0;

    if (std::abs(omega) < 1e-10) {
        F(0, 0) = 1.0;
        F(0, 3) = dt;
        F(1, 1) = 1.0;
        F(1, 4) = dt;
        F(3, 3) = 1.0;
        F(4, 4) = 1.0;
    } else {
        double s = std::sin(omega * dt);
        double c = std::cos(omega * dt);
        double so = s / omega;
        double co = (1.0 - c) / omega;

        F(0, 0) = 1.0;
        F(0, 3) = so;
        F(0, 4) = -co;
        F(0, 6) = (c * dt / omega - so / omega) * vx + (s * dt / omega - co / omega) * vy;

        F(1, 1) = 1.0;
        F(1, 3) = co;
        F(1, 4) = so;
        F(1, 6) = (s * dt / omega - co / omega) * vx + (-c * dt / omega + so / omega) * vy;

        F(3, 3) = c;
        F(3, 4) = -s;
        F(3, 6) = -s * dt * vx - c * dt * vy;

        F(4, 3) = s;
        F(4, 4) = c;
        F(4, 6) = c * dt * vx - s * dt * vy;
    }
    return F;
}

CoordinatedTurn::StateMat CoordinatedTurn::process_noise(double dt) const {
    double dt2 = dt * dt;
    double dt3 = dt2 * dt;

    StateMat Q = StateMat::Zero();
    // Position-velocity block (x, y, z and their velocities)
    for (int i = 0; i < 3; ++i) {
        Q(i, i) = q_ * dt3 / 3.0;
        Q(i, i + 3) = q_ * dt2 / 2.0;
        Q(i + 3, i) = q_ * dt2 / 2.0;
        Q(i + 3, i + 3) = q_ * dt;
    }
    // Turn rate process noise
    Q(6, 6) = q_ * dt;
    return Q;
}

} // namespace aot::motion
