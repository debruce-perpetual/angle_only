#pragma once

#include "angle_only/core/types.hpp"
#include "angle_only/core/state.hpp"
#include "angle_only/core/detection.hpp"
#include "angle_only/motion/constant_velocity_msc.hpp"
#include "angle_only/measurement/msc_measurement.hpp"
#include <vector>

namespace aot::filters {

/// Modified Spherical Coordinates Extended Kalman Filter.
/// Implements MATLAB's trackingMSCEKF: predict, correct, correct_jpda,
/// distance, likelihood, and RTS smoothing.
class MSCEKF {
public:
    MSCEKF();
    explicit MSCEKF(const MSCState& initial_state);

    // Core filter operations
    void predict(double dt);
    void correct(const Vec2& measurement, const Mat2& R);

    /// JPDA-style correction with weighted measurement
    void correct_jpda(const std::vector<Vec2>& measurements,
                      const std::vector<double>& weights,
                      const Mat2& R);

    /// Mahalanobis distance between predicted measurement and z
    [[nodiscard]] double distance(const Vec2& z, const Mat2& R) const;

    /// Measurement likelihood (Gaussian)
    [[nodiscard]] double likelihood(const Vec2& z, const Mat2& R) const;

    // State access
    [[nodiscard]] const Vec6& state() const { return state_.x; }
    [[nodiscard]] const Mat6& covariance() const { return state_.P; }
    [[nodiscard]] const MSCState& msc_state() const { return state_; }

    void set_state(const Vec6& x) { state_.x = x; }
    void set_covariance(const Mat6& P) { state_.P = P; }

    // Model access
    [[nodiscard]] motion::ConstantVelocityMSC& motion_model() { return motion_; }
    [[nodiscard]] const motion::ConstantVelocityMSC& motion_model() const { return motion_; }
    [[nodiscard]] measurement::MSCMeasurement& measurement_model() { return meas_; }
    [[nodiscard]] const measurement::MSCMeasurement& measurement_model() const { return meas_; }

    // History for smoothing
    struct FilterStep {
        Vec6 x_pred;   // predicted state
        Mat6 P_pred;   // predicted covariance
        Vec6 x_corr;   // corrected state
        Mat6 P_corr;   // corrected covariance
        Mat6 F;        // state transition Jacobian
    };

    /// RTS fixed-interval smoother over stored history
    [[nodiscard]] std::vector<Vec6> smooth() const;

    /// Enable/disable history storage for smoothing
    void set_store_history(bool store) { store_history_ = store; }
    void clear_history() { history_.clear(); }

private:
    MSCState state_;
    motion::ConstantVelocityMSC motion_;
    measurement::MSCMeasurement meas_;

    bool store_history_ = false;
    std::vector<FilterStep> history_;
};

/// Initialize MSCEKF from a detection (MATLAB's initcvmscekf)
[[nodiscard]] MSCEKF initcvmscekf(const Detection& det,
                                    double initial_inv_range = 0.01,
                                    double inv_range_std = 0.05);

} // namespace aot::filters
