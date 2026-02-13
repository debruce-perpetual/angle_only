#pragma once

#include "angle_only/core/types.hpp"
#include "angle_only/core/constants.hpp"

namespace aot::coords {

/// Spherical (az, el, r) to Cartesian (x, y, z)
/// Convention: az from X toward Y in XY plane, el from XY plane toward Z
[[nodiscard]] Vec3 spherical_to_cartesian(double az, double el, double r);

/// Cartesian to spherical (az, el, r)
struct SphericalCoords {
    double az;
    double el;
    double r;
};
[[nodiscard]] SphericalCoords cartesian_to_spherical(const Vec3& pos);

/// MSC (az, el, 1/r) to Cartesian position
[[nodiscard]] Vec3 msc_to_cartesian(double az, double el, double inv_range);

/// Cartesian to MSC (az, el, 1/r)
struct MSCCoords {
    double az;
    double el;
    double inv_range;
};
[[nodiscard]] MSCCoords cartesian_to_msc(const Vec3& pos);

/// Jacobian of MSC-to-Cartesian transform: d[x,y,z]/d[az,el,1/r]
[[nodiscard]] Mat3 msc_to_cartesian_jacobian(double az, double el, double inv_range);

/// Jacobian of Cartesian-to-MSC transform: d[az,el,1/r]/d[x,y,z]
[[nodiscard]] Mat3 cartesian_to_msc_jacobian(const Vec3& pos);

/// Unit direction vector from azimuth and elevation
[[nodiscard]] Vec3 az_el_to_unit_vector(double az, double el);

/// Azimuth and elevation from unit direction vector
struct AzEl {
    double az;
    double el;
};
[[nodiscard]] AzEl unit_vector_to_az_el(const Vec3& dir);

/// Rotation matrix: ENU (East-North-Up) to body frame
[[nodiscard]] Mat3 enu_to_body(double yaw, double pitch, double roll);

/// Rotation matrix: body frame to ENU
[[nodiscard]] Mat3 body_to_enu(double yaw, double pitch, double roll);

} // namespace aot::coords
