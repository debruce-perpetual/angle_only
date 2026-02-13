#include "angle_only/core/state.hpp"
#include <cmath>

namespace aot {

Vec3 msc_to_cartesian(double az, double el, double inv_range) {
    double r = 1.0 / inv_range;
    double cos_el = std::cos(el);
    return Vec3{r * cos_el * std::cos(az),
                r * cos_el * std::sin(az),
                r * std::sin(el)};
}

MSCComponents cartesian_to_msc(const Vec3& pos) {
    double r = pos.norm();
    return MSCComponents{
        std::atan2(pos.y(), pos.x()),
        std::asin(pos.z() / r),
        1.0 / r
    };
}

} // namespace aot
