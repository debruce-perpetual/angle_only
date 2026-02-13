#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "angle_only/core/state.hpp"
#include "angle_only/core/constants.hpp"
#include <cmath>

using namespace aot;
using Catch::Matchers::WithinAbs;

TEST_CASE("MSCState default construction", "[core][state]") {
    MSCState s;
    REQUIRE(s.x.isZero());
    REQUIRE(s.P.isApprox(Mat6::Identity()));
}

TEST_CASE("MSCState accessors", "[core][state]") {
    MSCState s;
    s.set_az(0.5);
    s.set_el(0.3);
    s.set_inv_range(0.01);
    s.set_az_rate(0.1);
    s.set_el_rate(-0.05);
    s.set_inv_range_rate(0.001);

    REQUIRE_THAT(s.az(), WithinAbs(0.5, 1e-15));
    REQUIRE_THAT(s.el(), WithinAbs(0.3, 1e-15));
    REQUIRE_THAT(s.inv_range(), WithinAbs(0.01, 1e-15));
    REQUIRE_THAT(s.az_rate(), WithinAbs(0.1, 1e-15));
    REQUIRE_THAT(s.el_rate(), WithinAbs(-0.05, 1e-15));
    REQUIRE_THAT(s.inv_range_rate(), WithinAbs(0.001, 1e-15));
}

TEST_CASE("CartesianState accessors", "[core][state]") {
    CartesianState s;
    s.set_position(Vec3{100.0, 200.0, 50.0});
    s.set_velocity(Vec3{10.0, -5.0, 1.0});

    REQUIRE(s.position().isApprox(Vec3{100.0, 200.0, 50.0}));
    REQUIRE(s.velocity().isApprox(Vec3{10.0, -5.0, 1.0}));
}

TEST_CASE("MSC to Cartesian conversion", "[core][state]") {
    // Known point: az=0, el=0, inv_range=0.01 => r=100, pos=(100,0,0)
    Vec3 pos = msc_to_cartesian(0.0, 0.0, 0.01);
    REQUIRE_THAT(pos.x(), WithinAbs(100.0, 1e-10));
    REQUIRE_THAT(pos.y(), WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(pos.z(), WithinAbs(0.0, 1e-10));
}

TEST_CASE("Cartesian to MSC conversion", "[core][state]") {
    Vec3 pos{100.0, 0.0, 0.0};
    auto msc = cartesian_to_msc(pos);
    REQUIRE_THAT(msc.az, WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(msc.el, WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(msc.inv_range, WithinAbs(0.01, 1e-10));
}

TEST_CASE("MSC<->Cartesian roundtrip", "[core][state]") {
    double az = 0.7, el = 0.3, inv_r = 0.005;
    Vec3 cart = msc_to_cartesian(az, el, inv_r);
    auto msc = cartesian_to_msc(cart);

    REQUIRE_THAT(msc.az, WithinAbs(az, 1e-10));
    REQUIRE_THAT(msc.el, WithinAbs(el, 1e-10));
    REQUIRE_THAT(msc.inv_range, WithinAbs(inv_r, 1e-10));
}
