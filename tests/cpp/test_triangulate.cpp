#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "angle_only/fusion/triangulate_los.hpp"
#include "angle_only/core/constants.hpp"
#include <cmath>

using namespace aot;
using namespace aot::fusion;
using Catch::Matchers::WithinAbs;

TEST_CASE("Triangulate two orthogonal LOS", "[fusion][triangulate]") {
    // Sensor 1 at origin looking along x-axis
    // Sensor 2 at (0,100,0) looking along x-axis (toward same target)
    // Target at (100, 50, 0)

    Vec3 target{100.0, 50.0, 0.0};

    LOSMeasurement m1;
    m1.origin = Vec3{0.0, 0.0, 0.0};
    m1.direction = (target - m1.origin).normalized();
    m1.noise = Mat2::Identity() * 1e-4;

    LOSMeasurement m2;
    m2.origin = Vec3{0.0, 100.0, 0.0};
    m2.direction = (target - m2.origin).normalized();
    m2.noise = Mat2::Identity() * 1e-4;

    auto result = triangulate_los({m1, m2});
    REQUIRE(result.valid);
    REQUIRE(result.position.isApprox(target, 1.0));  // within 1m
}

TEST_CASE("Triangulate needs at least 2 LOS", "[fusion][triangulate]") {
    LOSMeasurement m1;
    m1.origin = Vec3::Zero();
    m1.direction = Vec3{1.0, 0.0, 0.0};
    m1.noise = Mat2::Identity() * 1e-4;

    auto result = triangulate_los({m1});
    REQUIRE_FALSE(result.valid);
}

TEST_CASE("Triangulate with 3 sensors", "[fusion][triangulate]") {
    Vec3 target{50.0, 50.0, 20.0};

    std::vector<LOSMeasurement> measurements(3);
    Vec3 origins[] = {{0,0,0}, {100,0,0}, {0,100,0}};

    for (int i = 0; i < 3; ++i) {
        measurements[static_cast<size_t>(i)].origin = origins[i];
        measurements[static_cast<size_t>(i)].direction = (target - origins[i]).normalized();
        measurements[static_cast<size_t>(i)].noise = Mat2::Identity() * 1e-4;
    }

    auto result = triangulate_los(measurements);
    REQUIRE(result.valid);
    REQUIRE(result.position.isApprox(target, 0.1));  // tight tolerance with 3 sensors
}

TEST_CASE("Triangulate from sensors and detections", "[fusion][triangulate]") {
    Vec3 target{100.0, 0.0, 0.0};

    SensorPose s1;
    s1.position = Vec3{0.0, -50.0, 0.0};
    s1.orientation = Mat3::Identity();

    SensorPose s2;
    s2.position = Vec3{0.0, 50.0, 0.0};
    s2.orientation = Mat3::Identity();

    // Compute detections (az, el) from each sensor
    Vec3 dir1 = (target - s1.position).normalized();
    Vec3 dir2 = (target - s2.position).normalized();

    Detection d1;
    d1.azimuth = std::atan2(dir1.y(), dir1.x());
    d1.elevation = std::asin(dir1.z());
    d1.noise = Mat2::Identity() * 1e-4;

    Detection d2;
    d2.azimuth = std::atan2(dir2.y(), dir2.x());
    d2.elevation = std::asin(dir2.z());
    d2.noise = Mat2::Identity() * 1e-4;

    auto result = triangulate_los({s1, s2}, {d1, d2});
    REQUIRE(result.valid);
    REQUIRE(result.position.isApprox(target, 1.0));
}
