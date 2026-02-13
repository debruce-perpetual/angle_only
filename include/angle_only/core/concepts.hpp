#pragma once

#include "angle_only/core/types.hpp"
#include <concepts>

namespace aot {

/// A motion model provides state transition and its Jacobian
template <typename M>
concept MotionModel = requires(const M& m, const Vec6& state, double dt) {
    { m.predict(state, dt) } -> std::convertible_to<Vec6>;
    { m.jacobian(state, dt) } -> std::convertible_to<Mat6>;
    { m.process_noise(dt) } -> std::convertible_to<Mat6>;
    { M::state_dim } -> std::convertible_to<int>;
};

/// A measurement model maps state to measurements and provides Jacobian
template <typename M>
concept MeasurementModel = requires(const M& m, const Vec6& state) {
    { m.predict(state) } -> std::convertible_to<VecX>;
    { m.jacobian(state) } -> std::convertible_to<MatXR>;
    { m.noise() } -> std::convertible_to<MatXR>;
    { M::meas_dim } -> std::convertible_to<int>;
};

/// A tracking filter supports the standard predict/correct/distance/likelihood interface
template <typename F>
concept TrackingFilter = requires(F& f, double dt, const VecX& z, const MatXR& R) {
    { f.predict(dt) };
    { f.correct(z, R) };
    { f.state() } -> std::convertible_to<VecX>;
    { f.covariance() } -> std::convertible_to<MatXR>;
    { f.distance(z, R) } -> std::convertible_to<double>;
    { f.likelihood(z, R) } -> std::convertible_to<double>;
};

} // namespace aot
