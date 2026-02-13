#include "angle_only/measurement/msc_measurement.hpp"

namespace aot::measurement {

MSCMeasurement::MSCMeasurement(const Mat2& noise) : R_(noise) {}

Vec2 MSCMeasurement::predict(const Vec6& state) const {
    // Measurement is simply the first two components of the MSC state
    return state.head<2>();
}

Mat2x6 MSCMeasurement::jacobian(const Vec6& /*state*/) const {
    // H = [1 0 0 0 0 0]
    //     [0 1 0 0 0 0]
    Mat2x6 H = Mat2x6::Zero();
    H(0, 0) = 1.0;
    H(1, 1) = 1.0;
    return H;
}

} // namespace aot::measurement
