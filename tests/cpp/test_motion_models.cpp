#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "angle_only/motion/constant_velocity_msc.hpp"
#include "angle_only/motion/cv.hpp"
#include "angle_only/motion/ca.hpp"
#include "angle_only/motion/ct.hpp"
#include "angle_only/core/constants.hpp"
#include <cmath>

using namespace aot;
using Catch::Matchers::WithinAbs;

TEST_CASE("ConstantVelocityMSC prediction", "[motion][msc]") {
    motion::ConstantVelocityMSC model(1.0);
    Vec6 state;
    state << 0.5, 0.3, 0.01, 0.1, -0.05, 0.001;

    double dt = 1.0;
    Vec6 pred = model.predict(state, dt);

    REQUIRE_THAT(pred(0), WithinAbs(0.6, 1e-10));   // az + az_rate*dt
    REQUIRE_THAT(pred(1), WithinAbs(0.25, 1e-10));   // el + el_rate*dt
    REQUIRE_THAT(pred(2), WithinAbs(0.011, 1e-10));  // 1/r + inv_range_rate*dt
    REQUIRE_THAT(pred(3), WithinAbs(0.1, 1e-10));    // rates unchanged
    REQUIRE_THAT(pred(4), WithinAbs(-0.05, 1e-10));
    REQUIRE_THAT(pred(5), WithinAbs(0.001, 1e-10));
}

TEST_CASE("ConstantVelocityMSC Jacobian numerical check", "[motion][msc]") {
    motion::ConstantVelocityMSC model(1.0);
    Vec6 state;
    state << 0.5, 0.3, 0.01, 0.1, -0.05, 0.001;

    double dt = 0.5;
    Mat6 F = model.jacobian(state, dt);

    // Numerical Jacobian
    double eps = 1e-7;
    Vec6 f0 = model.predict(state, dt);
    Mat6 F_num;

    for (int i = 0; i < 6; ++i) {
        Vec6 s_pert = state;
        s_pert(i) += eps;
        Vec6 f1 = model.predict(s_pert, dt);
        F_num.col(i) = (f1 - f0) / eps;
    }

    REQUIRE(F.isApprox(F_num, 1e-5));
}

TEST_CASE("ConstantVelocityMSC process noise symmetry", "[motion][msc]") {
    motion::ConstantVelocityMSC model(1.0);
    Mat6 Q = model.process_noise(1.0);

    REQUIRE(Q.isApprox(Q.transpose(), 1e-15));
    // Positive semi-definite: all eigenvalues >= 0
    Eigen::SelfAdjointEigenSolver<Mat6> solver(Q);
    REQUIRE((solver.eigenvalues().array() >= -1e-15).all());
}

TEST_CASE("ConstantVelocity Cartesian prediction", "[motion][cv]") {
    motion::ConstantVelocity model(1.0);
    Vec6 state;
    state << 100.0, 200.0, 50.0, 10.0, -5.0, 1.0;

    Vec6 pred = model.predict(state, 2.0);
    REQUIRE_THAT(pred(0), WithinAbs(120.0, 1e-10));
    REQUIRE_THAT(pred(1), WithinAbs(190.0, 1e-10));
    REQUIRE_THAT(pred(2), WithinAbs(52.0, 1e-10));
}

TEST_CASE("ConstantAcceleration prediction", "[motion][ca]") {
    motion::ConstantAcceleration model(1.0);
    motion::ConstantAcceleration::State state;
    state << 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 1.0, 0.0, 0.0;

    auto pred = model.predict(state, 2.0);
    // x = 0 + 10*2 + 0.5*1*4 = 22
    REQUIRE_THAT(pred(0), WithinAbs(22.0, 1e-10));
    // vx = 10 + 1*2 = 12
    REQUIRE_THAT(pred(3), WithinAbs(12.0, 1e-10));
}

TEST_CASE("CoordinatedTurn zero omega degenerates to CV", "[motion][ct]") {
    motion::CoordinatedTurn model(1.0);
    motion::CoordinatedTurn::State state;
    state << 0.0, 0.0, 0.0, 10.0, 5.0, 1.0, 0.0;  // omega=0

    auto pred = model.predict(state, 1.0);
    REQUIRE_THAT(pred(0), WithinAbs(10.0, 1e-8));  // x + vx*dt
    REQUIRE_THAT(pred(1), WithinAbs(5.0, 1e-8));   // y + vy*dt
    REQUIRE_THAT(pred(2), WithinAbs(1.0, 1e-8));   // z + vz*dt
}

TEST_CASE("CoordinatedTurn 90-degree turn", "[motion][ct]") {
    motion::CoordinatedTurn model(1.0);
    double omega = half_pi;  // 90 deg/s
    motion::CoordinatedTurn::State state;
    state << 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, omega;

    auto pred = model.predict(state, 1.0);
    // After 90-degree turn: vx->0, vy->10
    REQUIRE_THAT(pred(3), WithinAbs(0.0, 1e-8));
    REQUIRE_THAT(pred(4), WithinAbs(10.0, 1e-8));
}
