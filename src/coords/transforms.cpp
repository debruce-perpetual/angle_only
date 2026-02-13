#include "angle_only/coords/transforms.hpp"
#include <cmath>
#include <stdexcept>

namespace aot::coords {

Vec3 spherical_to_cartesian(double az, double el, double r) {
    double cos_el = std::cos(el);
    return Vec3{r * cos_el * std::cos(az),
                r * cos_el * std::sin(az),
                r * std::sin(el)};
}

SphericalCoords cartesian_to_spherical(const Vec3& pos) {
    double r = pos.norm();
    if (r < 1e-15) {
        return {0.0, 0.0, 0.0};
    }
    return {std::atan2(pos.y(), pos.x()),
            std::asin(std::clamp(pos.z() / r, -1.0, 1.0)),
            r};
}

Vec3 msc_to_cartesian(double az, double el, double inv_range) {
    if (std::abs(inv_range) < 1e-15) {
        // At effectively infinite range, return unit direction
        double cos_el = std::cos(el);
        return Vec3{cos_el * std::cos(az),
                    cos_el * std::sin(az),
                    std::sin(el)} * 1e15;
    }
    double r = 1.0 / inv_range;
    double cos_el = std::cos(el);
    return Vec3{r * cos_el * std::cos(az),
                r * cos_el * std::sin(az),
                r * std::sin(el)};
}

MSCCoords cartesian_to_msc(const Vec3& pos) {
    double r = pos.norm();
    if (r < 1e-15) {
        return {0.0, 0.0, 1e15};
    }
    return {std::atan2(pos.y(), pos.x()),
            std::asin(std::clamp(pos.z() / r, -1.0, 1.0)),
            1.0 / r};
}

Mat3 msc_to_cartesian_jacobian(double az, double el, double inv_range) {
    // d[x,y,z] / d[az, el, 1/r]
    // x = cos(el)*cos(az) / inv_range
    // y = cos(el)*sin(az) / inv_range
    // z = sin(el) / inv_range
    double r = 1.0 / inv_range;
    double r2 = r * r;  // = 1/inv_range^2
    double ca = std::cos(az), sa = std::sin(az);
    double ce = std::cos(el), se = std::sin(el);

    Mat3 J;
    // dx/daz, dx/del, dx/d(1/r)
    J(0, 0) = -r * ce * sa;     // dx/daz
    J(0, 1) = -r * se * ca;     // dx/del
    J(0, 2) = -r2 * ce * ca;    // dx/d(1/r)

    // dy/daz, dy/del, dy/d(1/r)
    J(1, 0) = r * ce * ca;      // dy/daz
    J(1, 1) = -r * se * sa;     // dy/del
    J(1, 2) = -r2 * ce * sa;    // dy/d(1/r)

    // dz/daz, dz/del, dz/d(1/r)
    J(2, 0) = 0.0;              // dz/daz
    J(2, 1) = r * ce;           // dz/del
    J(2, 2) = -r2 * se;         // dz/d(1/r)

    return J;
}

Mat3 cartesian_to_msc_jacobian(const Vec3& pos) {
    double x = pos.x(), y = pos.y(), z = pos.z();
    double r2_xy = x * x + y * y;
    double r_xy = std::sqrt(r2_xy);
    double r2 = r2_xy + z * z;
    double r = std::sqrt(r2);
    double r3 = r * r2;

    Mat3 J;
    // daz/dx, daz/dy, daz/dz
    J(0, 0) = -y / r2_xy;
    J(0, 1) = x / r2_xy;
    J(0, 2) = 0.0;

    // del/dx, del/dy, del/dz
    J(1, 0) = -x * z / (r2 * r_xy);
    J(1, 1) = -y * z / (r2 * r_xy);
    J(1, 2) = r_xy / r2;

    // d(1/r)/dx, d(1/r)/dy, d(1/r)/dz
    J(2, 0) = -x / r3;
    J(2, 1) = -y / r3;
    J(2, 2) = -z / r3;

    return J;
}

Vec3 az_el_to_unit_vector(double az, double el) {
    double ce = std::cos(el);
    return Vec3{ce * std::cos(az), ce * std::sin(az), std::sin(el)};
}

AzEl unit_vector_to_az_el(const Vec3& dir) {
    Vec3 d = dir.normalized();
    return {std::atan2(d.y(), d.x()),
            std::asin(std::clamp(d.z(), -1.0, 1.0))};
}

Mat3 enu_to_body(double yaw, double pitch, double roll) {
    double cy = std::cos(yaw), sy = std::sin(yaw);
    double cp = std::cos(pitch), sp = std::sin(pitch);
    double cr = std::cos(roll), sr = std::sin(roll);

    Mat3 R;
    R(0, 0) = cy * cp;
    R(0, 1) = sy * cp;
    R(0, 2) = -sp;
    R(1, 0) = cy * sp * sr - sy * cr;
    R(1, 1) = sy * sp * sr + cy * cr;
    R(1, 2) = cp * sr;
    R(2, 0) = cy * sp * cr + sy * sr;
    R(2, 1) = sy * sp * cr - cy * sr;
    R(2, 2) = cp * cr;
    return R;
}

Mat3 body_to_enu(double yaw, double pitch, double roll) {
    // Transpose of enu_to_body (orthogonal matrix)
    Mat3 R = enu_to_body(yaw, pitch, roll);
    return R.transpose();
}

} // namespace aot::coords
