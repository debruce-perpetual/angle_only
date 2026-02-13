#include "angle_only/gpu/dispatch.hpp"
#include "angle_only/coords/angle_wrap.hpp"
#include <cmath>
#include <algorithm>

namespace aot::gpu {

void batch_predict(
    std::vector<Vec6>& states,
    std::vector<Mat6>& covariances,
    const Mat6& F,
    const Mat6& Q,
    double /*dt*/)
{
    for (size_t i = 0; i < states.size(); ++i) {
        states[i] = F * states[i];
        covariances[i] = F * covariances[i] * F.transpose() + Q;
    }
}

void batch_correct(
    std::vector<Vec6>& states,
    std::vector<Mat6>& covariances,
    const std::vector<Vec2>& measurements,
    const Mat2x6& H,
    const Mat2& R)
{
    for (size_t i = 0; i < states.size(); ++i) {
        Vec2 z_pred = H * states[i];
        Vec2 y = measurements[i] - z_pred;
        y(0) = coords::wrap_to_pi(y(0));

        Mat2 S = H * covariances[i] * H.transpose() + R;
        Eigen::Matrix<double, 6, 2> K = covariances[i] * H.transpose() * S.inverse();

        states[i] += K * y;
        states[i](0) = coords::wrap_to_pi(states[i](0));

        Mat6 I_KH = Mat6::Identity() - K * H;
        covariances[i] = I_KH * covariances[i] * I_KH.transpose() + K * R * K.transpose();
    }
}

void batch_likelihood(
    const std::vector<Vec6>& states,
    const std::vector<Mat6>& covariances,
    const std::vector<Vec2>& measurements,
    const Mat2x6& H,
    const Mat2& R,
    std::vector<double>& likelihoods)
{
    likelihoods.resize(states.size() * measurements.size());
    constexpr double log_2pi = 1.8378770664093453;

    for (size_t i = 0; i < states.size(); ++i) {
        Mat2 S = H * covariances[i] * H.transpose() + R;
        double det_S = S.determinant();
        Mat2 S_inv = S.inverse();

        for (size_t j = 0; j < measurements.size(); ++j) {
            Vec2 y = measurements[j] - H * states[i];
            y(0) = coords::wrap_to_pi(y(0));
            double mah2 = y.transpose() * S_inv * y;
            likelihoods[i * measurements.size() + j] =
                std::exp(-0.5 * (mah2 + 2.0 * log_2pi + std::log(det_S)));
        }
    }
}

void batch_triangulate(
    const std::vector<std::vector<Vec3>>& origins,
    const std::vector<std::vector<Vec3>>& directions,
    std::vector<Vec3>& positions,
    std::vector<bool>& valid)
{
    positions.resize(origins.size());
    valid.resize(origins.size());

    for (size_t k = 0; k < origins.size(); ++k) {
        const auto& origs = origins[k];
        const auto& dirs = directions[k];

        if (origs.size() < 2) {
            valid[k] = false;
            continue;
        }

        Mat3 A = Mat3::Zero();
        Vec3 b = Vec3::Zero();

        for (size_t i = 0; i < origs.size(); ++i) {
            Vec3 d = dirs[i].normalized();
            Mat3 proj = Mat3::Identity() - d * d.transpose();
            A += proj;
            b += proj * origs[i];
        }

        Eigen::JacobiSVD<Mat3> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        auto sv = svd.singularValues();
        if (sv(sv.size() - 1) < 1e-10 * sv(0)) {
            valid[k] = false;
            continue;
        }

        positions[k] = svd.solve(b);
        valid[k] = true;
    }
}

void batch_gating(
    const std::vector<Vec2>& predicted_measurements,
    const std::vector<Mat2>& innovation_covariances,
    const std::vector<Vec2>& measurements,
    double threshold,
    std::vector<std::vector<int>>& gated_indices)
{
    double threshold_sq = threshold * threshold;
    gated_indices.resize(predicted_measurements.size());

    for (size_t i = 0; i < predicted_measurements.size(); ++i) {
        gated_indices[i].clear();
        Mat2 S_inv = innovation_covariances[i].inverse();

        for (size_t j = 0; j < measurements.size(); ++j) {
            Vec2 y = measurements[j] - predicted_measurements[i];
            y(0) = coords::wrap_to_pi(y(0));
            double d2 = y.transpose() * S_inv * y;
            if (d2 <= threshold_sq) {
                gated_indices[i].push_back(static_cast<int>(j));
            }
        }
    }
}

} // namespace aot::gpu
