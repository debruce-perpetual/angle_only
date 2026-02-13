#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "angle_only/measurement/msc_measurement.hpp"
#include "angle_only/measurement/spherical_measurement.hpp"
#include <cmath>

using namespace aot;
using Catch::Matchers::WithinAbs;

TEST_CASE("MSCMeasurement extracts az/el from state", "[measurement][msc]") {
    measurement::MSCMeasurement model;
    Vec6 state;
    state << 0.5, 0.3, 0.01, 0.1, -0.05, 0.001;

    Vec2 z = model.predict(state);
    REQUIRE_THAT(z(0), WithinAbs(0.5, 1e-15));
    REQUIRE_THAT(z(1), WithinAbs(0.3, 1e-15));
}

TEST_CASE("MSCMeasurement Jacobian", "[measurement][msc]") {
    measurement::MSCMeasurement model;
    Vec6 state;
    state << 0.5, 0.3, 0.01, 0.1, -0.05, 0.001;

    Mat2x6 H = model.jacobian(state);

    // H should be [1 0 0 0 0 0; 0 1 0 0 0 0]
    REQUIRE_THAT(H(0, 0), WithinAbs(1.0, 1e-15));
    REQUIRE_THAT(H(1, 1), WithinAbs(1.0, 1e-15));
    REQUIRE_THAT(H(0, 2), WithinAbs(0.0, 1e-15));
    REQUIRE_THAT(H(1, 0), WithinAbs(0.0, 1e-15));
}

TEST_CASE("SphericalMeasurement from Cartesian state", "[measurement][spherical]") {
    measurement::SphericalMeasurement model;

    // Target along x-axis at 100m
    Vec6 state;
    state << 100.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    Vec2 z = model.predict(state);
    REQUIRE_THAT(z(0), WithinAbs(0.0, 1e-12));  // az=0
    REQUIRE_THAT(z(1), WithinAbs(0.0, 1e-12));  // el=0
}

TEST_CASE("SphericalMeasurement Jacobian numerical check", "[measurement][spherical]") {
    measurement::SphericalMeasurement model;
    Vec6 state;
    state << 100.0, 50.0, 30.0, 10.0, -5.0, 1.0;

    Mat2x6 H = model.jacobian(state);

    // Numerical Jacobian
    double eps = 1e-7;
    Vec2 h0 = model.predict(state);
    Mat2x6 H_num = Mat2x6::Zero();

    for (int i = 0; i < 6; ++i) {
        Vec6 s_pert = state;
        s_pert(i) += eps;
        Vec2 h1 = model.predict(s_pert);
        H_num.col(i) = (h1 - h0) / eps;
    }

    REQUIRE(H.isApprox(H_num, 1e-4));
}
