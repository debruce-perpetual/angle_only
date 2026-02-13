#pragma once

#include "mex.h"
#include <angle_only/core/types.hpp>
#include <string>
#include <vector>
#include <stdexcept>

namespace aot::mex {

// ---------------------------------------------------------------------------
// Error helpers
// ---------------------------------------------------------------------------

inline void check_nargs(int nrhs, int expected, const char* cmd) {
    if (nrhs < expected) {
        std::string msg = std::string(cmd) + " requires at least " +
                          std::to_string(expected) + " arguments";
        mexErrMsgIdAndTxt("aot:nargs", "%s", msg.c_str());
    }
}

inline void check_numeric(const mxArray* a, const char* name) {
    if (!mxIsDouble(a) || mxIsComplex(a)) {
        std::string msg = std::string(name) + " must be a real double array";
        mexErrMsgIdAndTxt("aot:type", "%s", msg.c_str());
    }
}

inline void check_scalar(const mxArray* a, const char* name) {
    if (!mxIsDouble(a) || mxIsComplex(a) || mxGetNumberOfElements(a) != 1) {
        std::string msg = std::string(name) + " must be a real double scalar";
        mexErrMsgIdAndTxt("aot:type", "%s", msg.c_str());
    }
}

// ---------------------------------------------------------------------------
// String conversion
// ---------------------------------------------------------------------------

inline std::string mx_to_string(const mxArray* a) {
    if (!mxIsChar(a)) {
        mexErrMsgIdAndTxt("aot:type", "Expected a string argument");
    }
    char* str = mxArrayToString(a);
    std::string result(str);
    mxFree(str);
    return result;
}

// ---------------------------------------------------------------------------
// Scalar conversion
// ---------------------------------------------------------------------------

inline double mx_to_scalar(const mxArray* a, const char* name = "argument") {
    check_scalar(a, name);
    return mxGetScalar(a);
}

inline uint64_t mx_to_handle(const mxArray* a) {
    if (!mxIsUint64(a) || mxGetNumberOfElements(a) != 1) {
        mexErrMsgIdAndTxt("aot:type", "Handle must be a uint64 scalar");
    }
    return *static_cast<uint64_t*>(mxGetData(a));
}

inline mxArray* scalar_to_mx(double val) {
    mxArray* out = mxCreateDoubleScalar(val);
    return out;
}

inline mxArray* handle_to_mx(uint64_t h) {
    mxArray* out = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    *static_cast<uint64_t*>(mxGetData(out)) = h;
    return out;
}

inline mxArray* bool_to_mx(bool val) {
    return mxCreateLogicalScalar(val);
}

// ---------------------------------------------------------------------------
// Fixed-size vector conversions
// ---------------------------------------------------------------------------

template <int N>
Eigen::Matrix<double, N, 1> mx_to_vec(const mxArray* a, const char* name = "vector") {
    check_numeric(a, name);
    if (mxGetNumberOfElements(a) != static_cast<size_t>(N)) {
        std::string msg = std::string(name) + " must have " +
                          std::to_string(N) + " elements";
        mexErrMsgIdAndTxt("aot:size", "%s", msg.c_str());
    }
    Eigen::Matrix<double, N, 1> v;
    const double* data = mxGetPr(a);
    for (int i = 0; i < N; ++i) {
        v(i) = data[i];
    }
    return v;
}

template <int N>
mxArray* vec_to_mx(const Eigen::Matrix<double, N, 1>& v) {
    mxArray* out = mxCreateDoubleMatrix(N, 1, mxREAL);
    double* data = mxGetPr(out);
    for (int i = 0; i < N; ++i) {
        data[i] = v(i);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Fixed-size matrix conversions (row-major Eigen <-> column-major MATLAB)
// ---------------------------------------------------------------------------

template <int Rows, int Cols>
MatRM<double, Rows, Cols> mx_to_mat(const mxArray* a, const char* name = "matrix") {
    check_numeric(a, name);
    if (mxGetM(a) != static_cast<size_t>(Rows) ||
        mxGetN(a) != static_cast<size_t>(Cols)) {
        std::string msg = std::string(name) + " must be " +
                          std::to_string(Rows) + "x" + std::to_string(Cols);
        mexErrMsgIdAndTxt("aot:size", "%s", msg.c_str());
    }
    MatRM<double, Rows, Cols> M;
    const double* data = mxGetPr(a);
    // MATLAB is column-major, Eigen row-major: transpose during copy
    for (int c = 0; c < Cols; ++c) {
        for (int r = 0; r < Rows; ++r) {
            M(r, c) = data[c * Rows + r];
        }
    }
    return M;
}

template <int Rows, int Cols>
mxArray* mat_to_mx(const MatRM<double, Rows, Cols>& M) {
    mxArray* out = mxCreateDoubleMatrix(Rows, Cols, mxREAL);
    double* data = mxGetPr(out);
    // Eigen row-major -> MATLAB column-major
    for (int c = 0; c < Cols; ++c) {
        for (int r = 0; r < Rows; ++r) {
            data[c * Rows + r] = M(r, c);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Dynamic matrix/vector conversions
// ---------------------------------------------------------------------------

inline VecX mx_to_vecx(const mxArray* a, const char* name = "vector") {
    check_numeric(a, name);
    size_t n = mxGetNumberOfElements(a);
    VecX v(n);
    const double* data = mxGetPr(a);
    for (size_t i = 0; i < n; ++i) {
        v(static_cast<Eigen::Index>(i)) = data[i];
    }
    return v;
}

inline mxArray* vecx_to_mx(const VecX& v) {
    auto n = v.size();
    mxArray* out = mxCreateDoubleMatrix(static_cast<size_t>(n), 1, mxREAL);
    double* data = mxGetPr(out);
    for (Eigen::Index i = 0; i < n; ++i) {
        data[i] = v(i);
    }
    return out;
}

inline MatXR mx_to_matxr(const mxArray* a, const char* name = "matrix") {
    check_numeric(a, name);
    auto rows = static_cast<Eigen::Index>(mxGetM(a));
    auto cols = static_cast<Eigen::Index>(mxGetN(a));
    MatXR M(rows, cols);
    const double* data = mxGetPr(a);
    // MATLAB column-major -> Eigen row-major
    for (Eigen::Index c = 0; c < cols; ++c) {
        for (Eigen::Index r = 0; r < rows; ++r) {
            M(r, c) = data[c * rows + r];
        }
    }
    return M;
}

inline mxArray* matxr_to_mx(const MatXR& M) {
    auto rows = M.rows();
    auto cols = M.cols();
    mxArray* out = mxCreateDoubleMatrix(
        static_cast<size_t>(rows), static_cast<size_t>(cols), mxREAL);
    double* data = mxGetPr(out);
    // Eigen row-major -> MATLAB column-major
    for (Eigen::Index c = 0; c < cols; ++c) {
        for (Eigen::Index r = 0; r < rows; ++r) {
            data[c * rows + r] = M(r, c);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// std::vector<double> conversions
// ---------------------------------------------------------------------------

inline std::vector<double> mx_to_double_vector(const mxArray* a,
                                                 const char* name = "vector") {
    check_numeric(a, name);
    size_t n = mxGetNumberOfElements(a);
    const double* data = mxGetPr(a);
    return std::vector<double>(data, data + n);
}

inline mxArray* double_vector_to_mx(const std::vector<double>& v) {
    mxArray* out = mxCreateDoubleMatrix(static_cast<size_t>(v.size()), 1, mxREAL);
    double* data = mxGetPr(out);
    for (size_t i = 0; i < v.size(); ++i) {
        data[i] = v[i];
    }
    return out;
}

// ---------------------------------------------------------------------------
// std::vector<int> conversions (returns as double for MATLAB)
// ---------------------------------------------------------------------------

inline mxArray* int_vector_to_mx(const std::vector<int>& v) {
    mxArray* out = mxCreateDoubleMatrix(static_cast<size_t>(v.size()), 1, mxREAL);
    double* data = mxGetPr(out);
    for (size_t i = 0; i < v.size(); ++i) {
        data[i] = static_cast<double>(v[i] + 1);  // 0-indexed C++ -> 1-indexed MATLAB
    }
    return out;
}

// ---------------------------------------------------------------------------
// std::vector<Vec2> from MATLAB Nx2 or 2xN matrix
// ---------------------------------------------------------------------------

inline std::vector<Vec2> mx_to_vec2_list(const mxArray* a,
                                          const char* name = "measurements") {
    check_numeric(a, name);
    size_t rows = mxGetM(a);
    size_t cols = mxGetN(a);
    const double* data = mxGetPr(a);

    std::vector<Vec2> result;
    if (cols == 2) {
        // Nx2 matrix (each row is a measurement)
        result.resize(rows);
        for (size_t i = 0; i < rows; ++i) {
            result[i](0) = data[i];           // column 0
            result[i](1) = data[rows + i];    // column 1
        }
    } else if (rows == 2) {
        // 2xN matrix (each column is a measurement)
        result.resize(cols);
        for (size_t j = 0; j < cols; ++j) {
            result[j](0) = data[2 * j];
            result[j](1) = data[2 * j + 1];
        }
    } else {
        mexErrMsgIdAndTxt("aot:size", "%s must be Nx2 or 2xN", name);
    }
    return result;
}

// ---------------------------------------------------------------------------
// std::vector<VecX> from MATLAB NxM matrix (each row is a vector)
// ---------------------------------------------------------------------------

inline std::vector<VecX> mx_to_vecx_list(const mxArray* a,
                                           const char* name = "vectors") {
    check_numeric(a, name);
    auto rows = static_cast<Eigen::Index>(mxGetM(a));
    auto cols = static_cast<Eigen::Index>(mxGetN(a));
    const double* data = mxGetPr(a);

    std::vector<VecX> result(rows);
    for (Eigen::Index i = 0; i < rows; ++i) {
        result[i].resize(cols);
        for (Eigen::Index j = 0; j < cols; ++j) {
            result[i](j) = data[j * rows + i];
        }
    }
    return result;
}

} // namespace aot::mex
