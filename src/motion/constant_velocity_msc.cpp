#include "angle_only/motion/constant_velocity_msc.hpp"
#include "angle_only/coords/angle_wrap.hpp"
#include <cmath>

namespace aot::motion {

ConstantVelocityMSC::ConstantVelocityMSC(double process_noise_intensity)
    : q_(process_noise_intensity) {}

Vec6 ConstantVelocityMSC::predict(const Vec6& state, double dt) const {
    // Constant velocity in MSC: position += velocity * dt
    Vec6 x_pred;
    x_pred(0) = state(0) + state(3) * dt;  // az
    x_pred(1) = state(1) + state(4) * dt;  // el
    x_pred(2) = state(2) + state(5) * dt;  // 1/r
    x_pred(3) = state(3);                   // az_rate (constant)
    x_pred(4) = state(4);                   // el_rate (constant)
    x_pred(5) = state(5);                   // inv_range_rate (constant)

    // Wrap angles
    x_pred(0) = coords::wrap_to_pi(x_pred(0));
    x_pred(1) = std::clamp(x_pred(1), -aot::half_pi, aot::half_pi);

    return x_pred;
}

Mat6 ConstantVelocityMSC::jacobian(const Vec6& /*state*/, double dt) const {
    // Jacobian of constant velocity model is simple:
    // F = [I3  dt*I3]
    //     [0    I3  ]
    Mat6 F = Mat6::Identity();
    F(0, 3) = dt;
    F(1, 4) = dt;
    F(2, 5) = dt;
    return F;
}

Mat6 ConstantVelocityMSC::process_noise(double dt) const {
    // Continuous white noise acceleration model
    // Q = q * [dt^3/3 * I3,  dt^2/2 * I3]
    //         [dt^2/2 * I3,  dt     * I3]
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
