#include "angle_only/measurement/spherical_measurement.hpp"
#include <cmath>
#include <algorithm>

namespace aot::measurement {

SphericalMeasurement::SphericalMeasurement(const Mat2& noise) : R_(noise) {}

Vec2 SphericalMeasurement::predict(const Vec6& state) const {
    double x = state(0), y = state(1), z = state(2);
    double r = std::sqrt(x * x + y * y + z * z);
    Vec2 z_pred;
    z_pred(0) = std::atan2(y, x);
    z_pred(1) = std::asin(std::clamp(z / r, -1.0, 1.0));
    return z_pred;
}

Mat2x6 SphericalMeasurement::jacobian(const Vec6& state) const {
    double x = state(0), y = state(1), z = state(2);
    double r2_xy = x * x + y * y;
    double r_xy = std::sqrt(r2_xy);
    double r2 = r2_xy + z * z;

    Mat2x6 H = Mat2x6::Zero();

    // daz/dx, daz/dy
    H(0, 0) = -y / r2_xy;
    H(0, 1) = x / r2_xy;

    // del/dx, del/dy, del/dz
    H(1, 0) = -x * z / (r2 * r_xy);
    H(1, 1) = -y * z / (r2 * r_xy);
    H(1, 2) = r_xy / r2;

    return H;
}

} // namespace aot::measurement
