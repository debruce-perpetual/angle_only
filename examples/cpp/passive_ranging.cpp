/// Passive ranging example: MSCEKF tracking a target using angle-only measurements.
#include <angle_only/angle_only.hpp>
#include <iostream>
#include <cmath>

int main() {
    using namespace aot;

    // Target: constant velocity, starting at (500, 200, 50) m
    Vec3 target_pos{500.0, 200.0, 50.0};
    Vec3 target_vel{10.0, -5.0, 0.5};
    double dt = 1.0;

    // Sensor at origin
    Vec3 sensor_pos = Vec3::Zero();
    Mat2 R = Mat2::Identity() * 1e-6;  // ~1 mrad noise

    // First detection
    Vec3 rel = target_pos - sensor_pos;
    Detection det;
    det.azimuth = std::atan2(rel.y(), rel.x());
    det.elevation = std::asin(rel.z() / rel.norm());
    det.noise = R;

    // Initialize MSCEKF
    auto ekf = filters::initcvmscekf(det);

    std::cout << "Passive Ranging Demo\n";
    std::cout << "Step | True Az   | Est Az    | True El   | Est El    | Est Range\n";
    std::cout << "-----|-----------|-----------|-----------|-----------|----------\n";

    for (int k = 0; k < 30; ++k) {
        target_pos += target_vel * dt;
        rel = target_pos - sensor_pos;
        double true_az = std::atan2(rel.y(), rel.x());
        double true_el = std::asin(rel.z() / rel.norm());
        double true_range = rel.norm();

        // Predict
        ekf.predict(dt);

        // Simulate measurement (no noise for clean demo)
        Vec2 z;
        z << true_az, true_el;
        ekf.correct(z, R);

        // Estimated range from inverse range
        double est_range = (ekf.state()(2) > 1e-10) ? 1.0 / ekf.state()(2) : -1.0;

        std::printf("%4d | %9.5f | %9.5f | %9.5f | %9.5f | %8.1f (true: %.1f)\n",
            k, true_az, ekf.state()(0), true_el, ekf.state()(1),
            est_range, true_range);
    }

    return 0;
}
