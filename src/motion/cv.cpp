#include "angle_only/motion/cv.hpp"

namespace aot::motion {

ConstantVelocity::ConstantVelocity(double process_noise_intensity)
    : q_(process_noise_intensity) {}

Vec6 ConstantVelocity::predict(const Vec6& state, double dt) const {
    Vec6 x_pred;
    x_pred.head<3>() = state.head<3>() + state.tail<3>() * dt;
    x_pred.tail<3>() = state.tail<3>();
    return x_pred;
}

Mat6 ConstantVelocity::jacobian(const Vec6& /*state*/, double dt) const {
    Mat6 F = Mat6::Identity();
    F(0, 3) = dt;
    F(1, 4) = dt;
    F(2, 5) = dt;
    return F;
}

Mat6 ConstantVelocity::process_noise(double dt) const {
    double dt2 = dt * dt;
    double dt3 = dt2 * dt;

    Mat6 Q = Mat6::Zero();
    for (int i = 0; i < 3; ++i) {
        Q(i, i) = q_ * dt3 / 3.0;
        Q(i, i + 3) = q_ * dt2 / 2.0;
        Q(i + 3, i) = q_ * dt2 / 2.0;
        Q(i + 3, i + 3) = q_ * dt;
    }
    return Q;
}

} // namespace aot::motion
