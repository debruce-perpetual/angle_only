#pragma once

#include "angle_only/core/constants.hpp"
#include <cmath>

namespace aot::coords {

/// Wrap angle to [-pi, pi]
[[nodiscard]] inline double wrap_to_pi(double angle) {
    angle = std::fmod(angle + pi, two_pi);
    if (angle < 0.0) angle += two_pi;
    return angle - pi;
}

/// Wrap angle to [0, 2*pi]
[[nodiscard]] inline double wrap_to_2pi(double angle) {
    angle = std::fmod(angle, two_pi);
    if (angle < 0.0) angle += two_pi;
    return angle;
}

/// Signed angular difference (a - b), wrapped to [-pi, pi]
[[nodiscard]] inline double angle_diff(double a, double b) {
    return wrap_to_pi(a - b);
}

} // namespace aot::coords
