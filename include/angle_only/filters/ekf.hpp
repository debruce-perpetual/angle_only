#pragma once

#include "angle_only/core/types.hpp"
#include <functional>

namespace aot::filters {

/// Generic Extended Kalman Filter.
/// Templated on state and measurement dimensions for compile-time sizing.
template <int StateDim, int MeasDim>
class EKF {
public:
    using StateVec = Eigen::Matrix<double, StateDim, 1>;
    using StateMat = Eigen::Matrix<double, StateDim, StateDim, Eigen::RowMajor>;
    using MeasVec = Eigen::Matrix<double, MeasDim, 1>;
    using MeasMat = Eigen::Matrix<double, MeasDim, MeasDim, Eigen::RowMajor>;
    using ObsMat = Eigen::Matrix<double, MeasDim, StateDim, Eigen::RowMajor>;
    using GainMat = Eigen::Matrix<double, StateDim, MeasDim>;

    using TransitionFn = std::function<StateVec(const StateVec&, double)>;
    using TransitionJacFn = std::function<StateMat(const StateVec&, double)>;
    using ProcessNoiseFn = std::function<StateMat(double)>;
    using MeasurementFn = std::function<MeasVec(const StateVec&)>;
    using MeasurementJacFn = std::function<ObsMat(const StateVec&)>;

    EKF() = default;

    void set_state(const StateVec& x) { x_ = x; }
    void set_covariance(const StateMat& P) { P_ = P; }

    void set_transition(TransitionFn f, TransitionJacFn jac, ProcessNoiseFn Q) {
        f_ = std::move(f);
        f_jac_ = std::move(jac);
        Q_fn_ = std::move(Q);
    }

    void set_measurement(MeasurementFn h, MeasurementJacFn jac) {
        h_ = std::move(h);
        h_jac_ = std::move(jac);
    }

    void predict(double dt) {
        StateMat F = f_jac_(x_, dt);
        x_ = f_(x_, dt);
        P_ = F * P_ * F.transpose() + Q_fn_(dt);
    }

    void correct(const MeasVec& z, const MeasMat& R) {
        ObsMat H = h_jac_(x_);
        MeasVec z_pred = h_(x_);
        MeasVec y = z - z_pred;
        MeasMat S = H * P_ * H.transpose() + R;
        GainMat K = P_ * H.transpose() * S.inverse();
        x_ += K * y;
        StateMat I_KH = StateMat::Identity() - K * H;
        P_ = I_KH * P_ * I_KH.transpose() + K * R * K.transpose();
    }

    [[nodiscard]] double distance(const MeasVec& z, const MeasMat& R) const {
        ObsMat H = h_jac_(x_);
        MeasVec y = z - h_(x_);
        MeasMat S = H * P_ * H.transpose() + R;
        return std::sqrt(static_cast<double>(y.transpose() * S.inverse() * y));
    }

    [[nodiscard]] double likelihood(const MeasVec& z, const MeasMat& R) const {
        ObsMat H = h_jac_(x_);
        MeasVec y = z - h_(x_);
        MeasMat S = H * P_ * H.transpose() + R;
        double det_S = S.determinant();
        double mah2 = y.transpose() * S.inverse() * y;
        constexpr double log_2pi = 1.8378770664093453;
        return std::exp(-0.5 * (mah2 + MeasDim * log_2pi + std::log(det_S)));
    }

    [[nodiscard]] const StateVec& state() const { return x_; }
    [[nodiscard]] const StateMat& covariance() const { return P_; }

private:
    StateVec x_ = StateVec::Zero();
    StateMat P_ = StateMat::Identity();
    TransitionFn f_;
    TransitionJacFn f_jac_;
    ProcessNoiseFn Q_fn_;
    MeasurementFn h_;
    MeasurementJacFn h_jac_;
};

} // namespace aot::filters
