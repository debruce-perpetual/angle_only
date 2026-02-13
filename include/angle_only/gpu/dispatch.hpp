#pragma once

#include "angle_only/core/types.hpp"
#include <vector>
#include <cstddef>

namespace aot::gpu {

/// Check if CUDA is available at runtime
[[nodiscard]] bool cuda_available();

/// Get the number of CUDA devices
[[nodiscard]] int device_count();

/// Batch size threshold above which GPU dispatch is beneficial
inline constexpr size_t gpu_batch_threshold = 64;

/// Should we dispatch to GPU for the given batch size?
[[nodiscard]] inline bool should_use_gpu(size_t batch_size) {
    return cuda_available() && batch_size >= gpu_batch_threshold;
}

// ---- Batch operations (auto-dispatch CPU/GPU) ----

/// Batch EKF predict: run predict on N filters simultaneously
void batch_predict(
    std::vector<Vec6>& states,
    std::vector<Mat6>& covariances,
    const Mat6& F,
    const Mat6& Q,
    double dt);

/// Batch EKF correct: run correct on N filters simultaneously
void batch_correct(
    std::vector<Vec6>& states,
    std::vector<Mat6>& covariances,
    const std::vector<Vec2>& measurements,
    const Mat2x6& H,
    const Mat2& R);

/// Batch likelihood computation
void batch_likelihood(
    const std::vector<Vec6>& states,
    const std::vector<Mat6>& covariances,
    const std::vector<Vec2>& measurements,
    const Mat2x6& H,
    const Mat2& R,
    std::vector<double>& likelihoods);

/// Batch triangulation: triangulate N target sets in parallel
void batch_triangulate(
    const std::vector<std::vector<Vec3>>& origins,
    const std::vector<std::vector<Vec3>>& directions,
    std::vector<Vec3>& positions,
    std::vector<bool>& valid);

/// Batch Mahalanobis gating
void batch_gating(
    const std::vector<Vec2>& predicted_measurements,
    const std::vector<Mat2>& innovation_covariances,
    const std::vector<Vec2>& measurements,
    double threshold,
    std::vector<std::vector<int>>& gated_indices);

} // namespace aot::gpu
