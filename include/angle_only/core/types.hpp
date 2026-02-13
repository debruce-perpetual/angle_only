#pragma once

#include <Eigen/Dense>
#include <cstddef>

namespace aot {

// Row-major for numpy interop (avoids copies in pybind11)
template <typename Scalar, int Rows, int Cols>
using MatRM = Eigen::Matrix<Scalar, Rows, Cols, Eigen::RowMajor>;

// Common fixed-size types
using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
using Vec4 = Eigen::Vector4d;
using Vec6 = Eigen::Matrix<double, 6, 1>;

using Mat2 = MatRM<double, 2, 2>;
using Mat3 = MatRM<double, 3, 3>;
using Mat4 = MatRM<double, 4, 4>;
using Mat6 = MatRM<double, 6, 6>;

// Row-major dynamic types (for Python interop)
using MatXR = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using VecX = Eigen::VectorXd;

// Fixed-size rectangular matrices common in tracking
using Mat6x3 = MatRM<double, 6, 3>;
using Mat3x6 = MatRM<double, 3, 6>;
using Mat6x2 = MatRM<double, 6, 2>;
using Mat2x6 = MatRM<double, 2, 6>;

// Index type
using Index = Eigen::Index;

} // namespace aot
