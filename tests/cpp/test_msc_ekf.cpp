#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "angle_only/filters/msc_ekf.hpp"
#include "angle_only/core/constants.hpp"
#include <cmath>

using namespace aot;
using namespace aot::filters;
using Catch::Matchers::WithinAbs;

TEST_CASE("MSCEKF initialization from detection", "[filters][mscekf]") {
    Detection det;
    det.azimuth = 0.5;
    det.elevation = 0.3;
    det.noise = Mat2::Identity() * 1e-4;

    MSCEKF ekf = initcvmscekf(det);

    REQUIRE_THAT(ekf.state()(0), WithinAbs(0.5, 1e-15));
    REQUIRE_THAT(ekf.state()(1), WithinAbs(0.3, 1e-15));
    REQUIRE_THAT(ekf.state()(2), WithinAbs(0.01, 1e-15));  // default inv_range
}

TEST_CASE("MSCEKF predict-correct cycle", "[filters][mscekf]") {
    Detection det;
    det.azimuth = 0.5;
    det.elevation = 0.3;
    det.noise = Mat2::Identity() * 1e-4;

    MSCEKF ekf = initcvmscekf(det);
    Mat2 R = Mat2::Identity() * 1e-4;

    // Predict
    ekf.predict(1.0);

    // Correct with measurement close to predicted
    Vec2 z;
    z << 0.51, 0.29;
    ekf.correct(z, R);

    // After correction, state should be pulled toward measurement
    REQUIRE(std::abs(ekf.state()(0) - 0.51) < 0.1);
    REQUIRE(std::abs(ekf.state()(1) - 0.29) < 0.1);
}

TEST_CASE("MSCEKF covariance shrinks after correction", "[filters][mscekf]") {
    Detection det;
    det.azimuth = 0.5;
    det.elevation = 0.3;
    det.noise = Mat2::Identity() * 1e-4;

    MSCEKF ekf = initcvmscekf(det);
    Mat2 R = Mat2::Identity() * 1e-4;

    ekf.predict(1.0);
    double trace_before = ekf.covariance().trace();

    Vec2 z;
    z << 0.5, 0.3;
    ekf.correct(z, R);
    double trace_after = ekf.covariance().trace();

    REQUIRE(trace_after < trace_before);
}

TEST_CASE("MSCEKF distance", "[filters][mscekf]") {
    Detection det;
    det.azimuth = 0.0;
    det.elevation = 0.0;
    det.noise = Mat2::Identity() * 1e-4;

    MSCEKF ekf = initcvmscekf(det);
    Mat2 R = Mat2::Identity() * 1e-4;

    // Distance to own prediction should be small
    Vec2 z;
    z << 0.0, 0.0;
    double d = ekf.distance(z, R);
    REQUIRE(d < 1.0);

    // Distance to far measurement should be large
    z << 1.0, 1.0;
    d = ekf.distance(z, R);
    REQUIRE(d > 10.0);
}

TEST_CASE("MSCEKF likelihood", "[filters][mscekf]") {
    Detection det;
    det.azimuth = 0.0;
    det.elevation = 0.0;
    det.noise = Mat2::Identity() * 1e-4;

    MSCEKF ekf = initcvmscekf(det);
    Mat2 R = Mat2::Identity() * 1e-4;

    Vec2 z_close, z_far;
    z_close << 0.001, 0.001;
    z_far << 1.0, 1.0;

    double lik_close = ekf.likelihood(z_close, R);
    double lik_far = ekf.likelihood(z_far, R);

    REQUIRE(lik_close > lik_far);
    REQUIRE(lik_close > 0.0);
}

TEST_CASE("MSCEKF RTS smoother", "[filters][mscekf]") {
    Detection det;
    det.azimuth = 0.0;
    det.elevation = 0.0;
    det.noise = Mat2::Identity() * 1e-4;

    MSCEKF ekf = initcvmscekf(det);
    ekf.set_store_history(true);
    Mat2 R = Mat2::Identity() * 1e-4;

    // Run a few steps
    for (int i = 0; i < 5; ++i) {
        ekf.predict(1.0);
        Vec2 z;
        z << 0.01 * (i + 1), 0.005 * (i + 1);
        ekf.correct(z, R);
    }

    auto smoothed = ekf.smooth();
    REQUIRE(smoothed.size() == 5);
}

TEST_CASE("MSCEKF JPDA correction", "[filters][mscekf]") {
    Detection det;
    det.azimuth = 0.0;
    det.elevation = 0.0;
    det.noise = Mat2::Identity() * 1e-4;

    MSCEKF ekf = initcvmscekf(det);
    Mat2 R = Mat2::Identity() * 1e-4;

    ekf.predict(1.0);

    std::vector<Vec2> measurements = {{Vec2{0.01, 0.005}}, {Vec2{0.02, 0.01}}};
    std::vector<double> weights = {0.7, 0.3};

    ekf.correct_jpda(measurements, weights, R);

    // Should not crash and state should be valid
    REQUIRE(std::isfinite(ekf.state()(0)));
    REQUIRE(std::isfinite(ekf.state()(1)));
}
