#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "angle_only/gpu/dispatch.hpp"
#include <cmath>

using namespace aot;
using namespace aot::gpu;
using Catch::Matchers::WithinAbs;

TEST_CASE("CPU batch predict", "[gpu][batch]") {
    std::vector<Vec6> states(3);
    std::vector<Mat6> covs(3);

    for (int i = 0; i < 3; ++i) {
        states[static_cast<size_t>(i)] = Vec6::Zero();
        states[static_cast<size_t>(i)](0) = 0.1 * (i + 1);
        covs[static_cast<size_t>(i)] = Mat6::Identity();
    }

    Mat6 F = Mat6::Identity();
    F(0, 3) = 1.0;
    F(1, 4) = 1.0;
    F(2, 5) = 1.0;
    Mat6 Q = Mat6::Identity() * 0.01;

    batch_predict(states, covs, F, Q, 1.0);

    // States should have been updated by F
    for (size_t i = 0; i < 3; ++i) {
        REQUIRE(std::isfinite(states[i](0)));
    }
}

TEST_CASE("CPU batch likelihood", "[gpu][batch]") {
    std::vector<Vec6> states(2);
    std::vector<Mat6> covs(2);
    states[0] = Vec6::Zero();
    states[1] = Vec6::Zero();
    states[1](0) = 1.0;
    covs[0] = Mat6::Identity();
    covs[1] = Mat6::Identity();

    std::vector<Vec2> measurements(1);
    measurements[0] = Vec2::Zero();

    Mat2x6 H = Mat2x6::Zero();
    H(0, 0) = 1.0;
    H(1, 1) = 1.0;
    Mat2 R = Mat2::Identity() * 0.01;

    std::vector<double> likelihoods;
    batch_likelihood(states, covs, measurements, H, R, likelihoods);

    REQUIRE(likelihoods.size() == 2);
    REQUIRE(likelihoods[0] > likelihoods[1]);  // closer state has higher likelihood
}

TEST_CASE("CPU batch triangulate", "[gpu][batch]") {
    Vec3 target{100.0, 50.0, 0.0};

    std::vector<std::vector<Vec3>> origins = {
        {Vec3{0,0,0}, Vec3{0,100,0}}
    };
    std::vector<std::vector<Vec3>> directions = {
        {(target - origins[0][0]).normalized(),
         (target - origins[0][1]).normalized()}
    };

    std::vector<Vec3> positions;
    std::vector<bool> valid;

    batch_triangulate(origins, directions, positions, valid);

    REQUIRE(valid.size() == 1);
    REQUIRE(valid[0]);
    REQUIRE(positions[0].isApprox(target, 1.0));
}

TEST_CASE("CPU batch gating", "[gpu][batch]") {
    std::vector<Vec2> pred_meas = {Vec2{0.0, 0.0}};
    std::vector<Mat2> innov_covs = {Mat2::Identity()};
    std::vector<Vec2> measurements = {
        Vec2{0.1, 0.0},   // close
        Vec2{10.0, 10.0}  // far
    };

    std::vector<std::vector<int>> gated;
    batch_gating(pred_meas, innov_covs, measurements, 3.0, gated);

    REQUIRE(gated.size() == 1);
    REQUIRE(gated[0].size() == 1);  // only close measurement passes
    REQUIRE(gated[0][0] == 0);
}

TEST_CASE("CUDA availability check", "[gpu]") {
    // Should not crash regardless of CUDA availability
    bool avail = cuda_available();
    int count = device_count();
    REQUIRE(count >= 0);
    if (!avail) {
        REQUIRE(count == 0);
    }
}
