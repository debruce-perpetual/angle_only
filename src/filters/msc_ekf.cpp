#include "angle_only/filters/msc_ekf.hpp"
#include "angle_only/coords/angle_wrap.hpp"
#include <cmath>
#include <numeric>

namespace aot::filters {

MSCEKF::MSCEKF() = default;

MSCEKF::MSCEKF(const MSCState& initial_state) : state_(initial_state) {}

void MSCEKF::predict(double dt) {
    Mat6 F = motion_.jacobian(state_.x, dt);
    Mat6 Q = motion_.process_noise(dt);

    Vec6 x_pred = motion_.predict(state_.x, dt);
    Mat6 P_pred = F * state_.P * F.transpose() + Q;

    if (store_history_) {
        history_.push_back({x_pred, P_pred, state_.x, state_.P, F});
    }

    state_.x = x_pred;
    state_.P = P_pred;
}

void MSCEKF::correct(const Vec2& measurement, const Mat2& R) {
    Mat2x6 H = meas_.jacobian(state_.x);
    Vec2 z_pred = meas_.predict(state_.x);

    // Innovation
    Vec2 y = measurement - z_pred;
    y(0) = coords::wrap_to_pi(y(0));  // wrap azimuth innovation

    // Innovation covariance
    Mat2 S = H * state_.P * H.transpose() + R;

    // Kalman gain
    // K = P * H' * S^{-1}, shape 6x2
    Eigen::Matrix<double, 6, 2> K = state_.P * H.transpose() * S.inverse();

    // State update
    state_.x += K * y;
    state_.x(0) = coords::wrap_to_pi(state_.x(0));

    // Joseph form covariance update for numerical stability
    Mat6 I_KH = Mat6::Identity() - K * H;
    state_.P = I_KH * state_.P * I_KH.transpose() + K * R * K.transpose();

    // Update history with corrected state
    if (store_history_ && !history_.empty()) {
        history_.back().x_corr = state_.x;
        history_.back().P_corr = state_.P;
    }
}

void MSCEKF::correct_jpda(const std::vector<Vec2>& measurements,
                            const std::vector<double>& weights,
                            const Mat2& R) {
    Mat2x6 H = meas_.jacobian(state_.x);
    Vec2 z_pred = meas_.predict(state_.x);

    // Innovation covariance
    Mat2 S = H * state_.P * H.transpose() + R;
    Eigen::Matrix<double, 6, 2> K = state_.P * H.transpose() * S.inverse();

    // Weighted innovation
    Vec2 y_combined = Vec2::Zero();
    Mat2 spread = Mat2::Zero();
    for (size_t i = 0; i < measurements.size(); ++i) {
        Vec2 y_i = measurements[i] - z_pred;
        y_i(0) = coords::wrap_to_pi(y_i(0));
        y_combined += weights[i] * y_i;
        spread += weights[i] * y_i * y_i.transpose();
    }

    // State update with combined innovation
    state_.x += K * y_combined;
    state_.x(0) = coords::wrap_to_pi(state_.x(0));

    // JPDA covariance update (accounts for measurement origin uncertainty)
    Mat2 P_spread = spread - y_combined * y_combined.transpose();
    Mat6 I_KH = Mat6::Identity() - K * H;
    state_.P = I_KH * state_.P * I_KH.transpose()
             + K * R * K.transpose()
             + K * P_spread * K.transpose();

    if (store_history_ && !history_.empty()) {
        history_.back().x_corr = state_.x;
        history_.back().P_corr = state_.P;
    }
}

double MSCEKF::distance(const Vec2& z, const Mat2& R) const {
    Mat2x6 H = meas_.jacobian(state_.x);
    Vec2 z_pred = meas_.predict(state_.x);

    Vec2 y = z - z_pred;
    y(0) = coords::wrap_to_pi(y(0));

    Mat2 S = H * state_.P * H.transpose() + R;
    return std::sqrt(y.transpose() * S.inverse() * y);
}

double MSCEKF::likelihood(const Vec2& z, const Mat2& R) const {
    Mat2x6 H = meas_.jacobian(state_.x);
    Vec2 z_pred = meas_.predict(state_.x);

    Vec2 y = z - z_pred;
    y(0) = coords::wrap_to_pi(y(0));

    Mat2 S = H * state_.P * H.transpose() + R;
    double det_S = S.determinant();
    double mah2 = y.transpose() * S.inverse() * y;

    constexpr double log_2pi = 1.8378770664093453;
    return std::exp(-0.5 * (mah2 + 2.0 * log_2pi + std::log(det_S)));
}

std::vector<Vec6> MSCEKF::smooth() const {
    if (history_.empty()) return {};

    auto n = history_.size();
    std::vector<Vec6> smoothed(n);
    smoothed[n - 1] = history_[n - 1].x_corr;

    // RTS backward pass
    for (int k = static_cast<int>(n) - 2; k >= 0; --k) {
        const auto& step = history_[static_cast<size_t>(k)];
        const auto& next = history_[static_cast<size_t>(k + 1)];

        Mat6 C = step.P_corr * step.F.transpose() * next.P_pred.inverse();
        smoothed[static_cast<size_t>(k)] = step.x_corr + C * (smoothed[static_cast<size_t>(k + 1)] - next.x_pred);
    }

    return smoothed;
}

MSCEKF initcvmscekf(const Detection& det, double initial_inv_range, double inv_range_std) {
    MSCState state;
    state.x(0) = det.azimuth;
    state.x(1) = det.elevation;
    state.x(2) = initial_inv_range;
    // Rates initialized to zero

    state.P = Mat6::Zero();
    state.P(0, 0) = det.noise(0, 0);
    state.P(1, 1) = det.noise(1, 1);
    state.P(2, 2) = inv_range_std * inv_range_std;
    // Rate uncertainties — large initial values
    state.P(3, 3) = 1.0;
    state.P(4, 4) = 1.0;
    state.P(5, 5) = 0.01;

    MSCEKF ekf(state);
    ekf.measurement_model().set_noise(det.noise);
    return ekf;
}

} // namespace aot::filters
