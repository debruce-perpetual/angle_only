#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "angle_only/coords/transforms.hpp"
#include "angle_only/coords/angle_wrap.hpp"
#include "angle_only/coords/frames.hpp"
#include "angle_only/core/constants.hpp"
#include <cmath>

using namespace aot;
using namespace aot::coords;
using Catch::Matchers::WithinAbs;

TEST_CASE("Spherical to Cartesian", "[coords][transforms]") {
    // Along x-axis: az=0, el=0, r=1
    Vec3 p = spherical_to_cartesian(0.0, 0.0, 1.0);
    REQUIRE_THAT(p.x(), WithinAbs(1.0, 1e-15));
    REQUIRE_THAT(p.y(), WithinAbs(0.0, 1e-15));
    REQUIRE_THAT(p.z(), WithinAbs(0.0, 1e-15));

    // Along y-axis: az=pi/2, el=0, r=1
    p = spherical_to_cartesian(half_pi, 0.0, 1.0);
    REQUIRE_THAT(p.x(), WithinAbs(0.0, 1e-15));
    REQUIRE_THAT(p.y(), WithinAbs(1.0, 1e-15));
    REQUIRE_THAT(p.z(), WithinAbs(0.0, 1e-15));

    // Along z-axis: az=0, el=pi/2, r=1
    p = spherical_to_cartesian(0.0, half_pi, 1.0);
    REQUIRE_THAT(p.x(), WithinAbs(0.0, 1e-15));
    REQUIRE_THAT(p.y(), WithinAbs(0.0, 1e-15));
    REQUIRE_THAT(p.z(), WithinAbs(1.0, 1e-15));
}

TEST_CASE("Cartesian to Spherical roundtrip", "[coords][transforms]") {
    Vec3 pos{3.0, 4.0, 5.0};
    auto sph = cartesian_to_spherical(pos);
    Vec3 recovered = spherical_to_cartesian(sph.az, sph.el, sph.r);
    REQUIRE(recovered.isApprox(pos, 1e-12));
}

TEST_CASE("MSC coordinate transforms", "[coords][transforms]") {
    double az = 1.2, el = 0.4, inv_r = 0.02;
    Vec3 cart = msc_to_cartesian(az, el, inv_r);
    auto msc = cartesian_to_msc(cart);

    REQUIRE_THAT(msc.az, WithinAbs(az, 1e-12));
    REQUIRE_THAT(msc.el, WithinAbs(el, 1e-12));
    REQUIRE_THAT(msc.inv_range, WithinAbs(inv_r, 1e-12));
}

TEST_CASE("MSC Jacobian numerical check", "[coords][transforms]") {
    double az = 0.5, el = 0.3, inv_r = 0.01;
    Mat3 J = msc_to_cartesian_jacobian(az, el, inv_r);

    // Numerical Jacobian
    double eps = 1e-7;
    Mat3 J_num;
    Vec3 f0 = msc_to_cartesian(az, el, inv_r);

    Vec3 f1 = msc_to_cartesian(az + eps, el, inv_r);
    J_num.col(0) = (f1 - f0) / eps;

    f1 = msc_to_cartesian(az, el + eps, inv_r);
    J_num.col(1) = (f1 - f0) / eps;

    f1 = msc_to_cartesian(az, el, inv_r + eps);
    J_num.col(2) = (f1 - f0) / eps;

    REQUIRE(J.isApprox(J_num, 1e-5));
}

TEST_CASE("Angle wrapping", "[coords][angle_wrap]") {
    REQUIRE_THAT(wrap_to_pi(0.0), WithinAbs(0.0, 1e-15));
    // pi and -pi are equivalent wrappings
    REQUIRE(std::abs(std::abs(wrap_to_pi(pi)) - pi) < 1e-15);
    REQUIRE(std::abs(std::abs(wrap_to_pi(-pi)) - pi) < 1e-15);
    REQUIRE(std::abs(std::abs(wrap_to_pi(3.0 * pi)) - pi) < 1e-10);
    REQUIRE(std::abs(std::abs(wrap_to_pi(-3.0 * pi)) - pi) < 1e-10);

    REQUIRE_THAT(wrap_to_2pi(0.0), WithinAbs(0.0, 1e-15));
    REQUIRE_THAT(wrap_to_2pi(-0.5), WithinAbs(two_pi - 0.5, 1e-15));
}

TEST_CASE("Az/El to unit vector roundtrip", "[coords][transforms]") {
    double az = 0.8, el = 0.4;
    Vec3 dir = az_el_to_unit_vector(az, el);
    REQUIRE_THAT(dir.norm(), WithinAbs(1.0, 1e-15));

    auto ae = unit_vector_to_az_el(dir);
    REQUIRE_THAT(ae.az, WithinAbs(az, 1e-12));
    REQUIRE_THAT(ae.el, WithinAbs(el, 1e-12));
}

TEST_CASE("Frame transforms roundtrip", "[coords][frames]") {
    Vec3 sensor_pos{100.0, 200.0, 50.0};
    Mat3 orientation = Mat3::Identity();

    Vec3 world_pos{150.0, 250.0, 75.0};
    Vec3 sensor_frame = world_to_sensor(world_pos, sensor_pos, orientation);
    Vec3 recovered = sensor_to_world(sensor_frame, sensor_pos, orientation);

    REQUIRE(recovered.isApprox(world_pos, 1e-12));
}
