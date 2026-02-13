#pragma once

/// @file angle_only.hpp
/// @brief Single-include convenience header for the angle-only tracking library.
/// @code
///   #include <angle_only/angle_only.hpp>
///   // All aot:: types, filters, models, fusion, and association are available.
/// @endcode

// Core types and interfaces
#include "angle_only/core/types.hpp"
#include "angle_only/core/constants.hpp"
#include "angle_only/core/state.hpp"
#include "angle_only/core/detection.hpp"
#include "angle_only/core/concepts.hpp"

// Coordinate transforms
#include "angle_only/coords/transforms.hpp"
#include "angle_only/coords/angle_wrap.hpp"
#include "angle_only/coords/frames.hpp"

// Motion models
#include "angle_only/motion/constant_velocity_msc.hpp"
#include "angle_only/motion/cv.hpp"
#include "angle_only/motion/ca.hpp"
#include "angle_only/motion/ct.hpp"

// Measurement models
#include "angle_only/measurement/msc_measurement.hpp"
#include "angle_only/measurement/spherical_measurement.hpp"

// Filters
#include "angle_only/filters/msc_ekf.hpp"
#include "angle_only/filters/ekf.hpp"
#include "angle_only/filters/gmphd.hpp"

// Fusion
#include "angle_only/fusion/triangulate_los.hpp"
#include "angle_only/fusion/static_detection_fuser.hpp"

// Association
#include "angle_only/association/gating.hpp"
#include "angle_only/association/gnn.hpp"
#include "angle_only/association/jpda.hpp"

// GPU (dispatch layer — works with or without CUDA)
#include "angle_only/gpu/dispatch.hpp"
