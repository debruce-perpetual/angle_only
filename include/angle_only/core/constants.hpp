#pragma once

#include <cmath>
#include <numbers>

namespace aot {

inline constexpr double pi = std::numbers::pi;
inline constexpr double two_pi = 2.0 * pi;
inline constexpr double half_pi = pi / 2.0;
inline constexpr double deg2rad = pi / 180.0;
inline constexpr double rad2deg = 180.0 / pi;

// Default filter parameters
inline constexpr double default_gate_threshold = 9.21;  // chi2(2, 0.99)
inline constexpr double default_merge_threshold = 4.0;
inline constexpr int default_max_components = 100;
inline constexpr double default_prune_threshold = 1e-5;

} // namespace aot
