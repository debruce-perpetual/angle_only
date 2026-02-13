#include "angle_only/fusion/triangulate_los.hpp"
#include "angle_only/coords/transforms.hpp"
#include <cmath>

namespace aot::fusion {

TriangulationResult triangulate_los(const std::vector<LOSMeasurement>& measurements) {
    TriangulationResult result;

    if (measurements.size() < 2) {
        return result;  // need at least 2 LOS
    }

    auto n = static_cast<Eigen::Index>(measurements.size());

    // Linear least-squares: minimize sum of squared distances from LOS lines
    // For each LOS: origin_i + t_i * dir_i
    // (I - d_i*d_i') * (p - o_i) = 0
    // Sum over i: A*p = b where A = sum(I - d*d'), b = sum((I - d*d')*o)

    Mat3 A = Mat3::Zero();
    Vec3 b = Vec3::Zero();

    for (Eigen::Index i = 0; i < n; ++i) {
        const auto& m = measurements[static_cast<size_t>(i)];
        Vec3 d = m.direction.normalized();
        Mat3 proj = Mat3::Identity() - d * d.transpose();
        A += proj;
        b += proj * m.origin;
    }

    // Solve A*p = b
    Eigen::JacobiSVD<Mat3> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

    // Check if problem is well-conditioned
    auto sv = svd.singularValues();
    if (sv(sv.size() - 1) < 1e-10 * sv(0)) {
        return result;  // degenerate geometry
    }

    result.position = svd.solve(b);

    // Compute residual (sum of squared distances to each LOS)
    double residual = 0.0;
    for (Eigen::Index i = 0; i < n; ++i) {
        const auto& m = measurements[static_cast<size_t>(i)];
        Vec3 d = m.direction.normalized();
        Vec3 diff = result.position - m.origin;
        Vec3 perp = diff - d * d.dot(diff);
        residual += perp.squaredNorm();
    }
    result.residual = residual;

    // Approximate covariance from pseudo-inverse
    result.covariance = (A.transpose() * A).inverse();
    result.valid = true;

    return result;
}

TriangulationResult triangulate_los(
    const std::vector<SensorPose>& sensors,
    const std::vector<Detection>& detections)
{
    std::vector<LOSMeasurement> los;
    los.reserve(detections.size());

    auto n = std::min(sensors.size(), detections.size());
    for (size_t i = 0; i < n; ++i) {
        LOSMeasurement m;
        m.origin = sensors[i].position;
        m.direction = sensors[i].orientation *
            coords::az_el_to_unit_vector(detections[i].azimuth, detections[i].elevation);
        m.noise = detections[i].noise;
        m.time = detections[i].time;
        m.sensor_id = sensors[i].sensor_id;
        los.push_back(m);
    }

    return triangulate_los(los);
}

} // namespace aot::fusion
