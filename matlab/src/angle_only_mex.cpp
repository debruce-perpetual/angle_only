/// @file angle_only_mex.cpp
/// @brief Single MEX gateway for the angle_only library.
/// All C++ objects are managed via integer handles (uint64).
/// MATLAB wrapper classes call: angle_only_mex('command', handle, ...)

#include "mex.h"
#include "mex_utils.hpp"
#include "object_store.hpp"

#include <angle_only/angle_only.hpp>

#include <string>
#include <unordered_map>
#include <functional>

using namespace aot;
using namespace aot::mex;

// Object stores for stateful types
static auto& ekf_store()  { return ObjectStore<filters::MSCEKF>::instance(); }
static auto& gmphd_store() { return ObjectStore<filters::GMPHD>::instance(); }

// Cleanup on mexAtExit
static void cleanup() {
    ekf_store().clear();
    gmphd_store().clear();
}

// ---------------------------------------------------------------------------
// MSCEKF commands
// ---------------------------------------------------------------------------

static void mscekf_create(int nlhs, mxArray* plhs[],
                           int nrhs, const mxArray* prhs[]) {
    uint64_t h = ekf_store().create();
    plhs[0] = handle_to_mx(h);
}

static void mscekf_create_from_detection(int nlhs, mxArray* plhs[],
                                          int nrhs, const mxArray* prhs[]) {
    // args: azimuth, elevation, noise(2x2), time, sensor_id, inv_range, inv_range_std
    check_nargs(nrhs, 4, "mscekf_create_from_detection");

    Detection det;
    det.azimuth = mx_to_scalar(prhs[1], "azimuth");
    det.elevation = mx_to_scalar(prhs[2], "elevation");
    det.noise = mx_to_mat<2, 2>(prhs[3], "noise");

    double inv_range = 0.01;
    double inv_range_std = 0.05;
    if (nrhs > 4) det.time = mx_to_scalar(prhs[4], "time");
    if (nrhs > 5) det.sensor_id = static_cast<uint32_t>(mx_to_scalar(prhs[5], "sensor_id"));
    if (nrhs > 6) inv_range = mx_to_scalar(prhs[6], "inv_range");
    if (nrhs > 7) inv_range_std = mx_to_scalar(prhs[7], "inv_range_std");

    auto ekf = filters::initcvmscekf(det, inv_range, inv_range_std);
    auto ptr = std::make_unique<filters::MSCEKF>(std::move(ekf));
    uint64_t h = ekf_store().store(std::move(ptr));
    plhs[0] = handle_to_mx(h);
}

static void mscekf_destroy(int nlhs, mxArray* plhs[],
                            int nrhs, const mxArray* prhs[]) {
    check_nargs(nrhs, 2, "mscekf_destroy");
    uint64_t h = mx_to_handle(prhs[1]);
    ekf_store().destroy(h);
}

static void mscekf_predict(int nlhs, mxArray* plhs[],
                            int nrhs, const mxArray* prhs[]) {
    check_nargs(nrhs, 3, "mscekf_predict");
    auto* ekf = ekf_store().get(mx_to_handle(prhs[1]));
    double dt = mx_to_scalar(prhs[2], "dt");
    ekf->predict(dt);
}

static void mscekf_correct(int nlhs, mxArray* plhs[],
                            int nrhs, const mxArray* prhs[]) {
    check_nargs(nrhs, 4, "mscekf_correct");
    auto* ekf = ekf_store().get(mx_to_handle(prhs[1]));
    Vec2 z = mx_to_vec<2>(prhs[2], "measurement");
    Mat2 R = mx_to_mat<2, 2>(prhs[3], "R");
    ekf->correct(z, R);
}

static void mscekf_correct_jpda(int nlhs, mxArray* plhs[],
                                 int nrhs, const mxArray* prhs[]) {
    check_nargs(nrhs, 5, "mscekf_correct_jpda");
    auto* ekf = ekf_store().get(mx_to_handle(prhs[1]));
    auto measurements = mx_to_vec2_list(prhs[2], "measurements");
    auto weights = mx_to_double_vector(prhs[3], "weights");
    Mat2 R = mx_to_mat<2, 2>(prhs[4], "R");
    ekf->correct_jpda(measurements, weights, R);
}

static void mscekf_state(int nlhs, mxArray* plhs[],
                          int nrhs, const mxArray* prhs[]) {
    check_nargs(nrhs, 2, "mscekf_state");
    auto* ekf = ekf_store().get(mx_to_handle(prhs[1]));
    plhs[0] = vec_to_mx<6>(ekf->state());
}

static void mscekf_covariance(int nlhs, mxArray* plhs[],
                               int nrhs, const mxArray* prhs[]) {
    check_nargs(nrhs, 2, "mscekf_covariance");
    auto* ekf = ekf_store().get(mx_to_handle(prhs[1]));
    plhs[0] = mat_to_mx<6, 6>(ekf->covariance());
}

static void mscekf_distance(int nlhs, mxArray* plhs[],
                             int nrhs, const mxArray* prhs[]) {
    check_nargs(nrhs, 4, "mscekf_distance");
    auto* ekf = ekf_store().get(mx_to_handle(prhs[1]));
    Vec2 z = mx_to_vec<2>(prhs[2], "measurement");
    Mat2 R = mx_to_mat<2, 2>(prhs[3], "R");
    plhs[0] = scalar_to_mx(ekf->distance(z, R));
}

static void mscekf_likelihood(int nlhs, mxArray* plhs[],
                               int nrhs, const mxArray* prhs[]) {
    check_nargs(nrhs, 4, "mscekf_likelihood");
    auto* ekf = ekf_store().get(mx_to_handle(prhs[1]));
    Vec2 z = mx_to_vec<2>(prhs[2], "measurement");
    Mat2 R = mx_to_mat<2, 2>(prhs[3], "R");
    plhs[0] = scalar_to_mx(ekf->likelihood(z, R));
}

static void mscekf_smooth(int nlhs, mxArray* plhs[],
                           int nrhs, const mxArray* prhs[]) {
    check_nargs(nrhs, 2, "mscekf_smooth");
    auto* ekf = ekf_store().get(mx_to_handle(prhs[1]));
    auto smoothed = ekf->smooth();

    // Return as Nx6 matrix
    size_t N = smoothed.size();
    mxArray* out = mxCreateDoubleMatrix(N, 6, mxREAL);
    double* data = mxGetPr(out);
    for (size_t i = 0; i < N; ++i) {
        for (int j = 0; j < 6; ++j) {
            data[j * N + i] = smoothed[i](j);
        }
    }
    plhs[0] = out;
}

static void mscekf_set_store_history(int nlhs, mxArray* plhs[],
                                      int nrhs, const mxArray* prhs[]) {
    check_nargs(nrhs, 3, "mscekf_set_store_history");
    auto* ekf = ekf_store().get(mx_to_handle(prhs[1]));
    bool store = mxIsLogicalScalarTrue(prhs[2]) ||
                 (mxIsDouble(prhs[2]) && mxGetScalar(prhs[2]) != 0.0);
    ekf->set_store_history(store);
}

static void mscekf_set_state(int nlhs, mxArray* plhs[],
                              int nrhs, const mxArray* prhs[]) {
    check_nargs(nrhs, 3, "mscekf_set_state");
    auto* ekf = ekf_store().get(mx_to_handle(prhs[1]));
    Vec6 x = mx_to_vec<6>(prhs[2], "state");
    ekf->set_state(x);
}

static void mscekf_set_covariance(int nlhs, mxArray* plhs[],
                                   int nrhs, const mxArray* prhs[]) {
    check_nargs(nrhs, 3, "mscekf_set_covariance");
    auto* ekf = ekf_store().get(mx_to_handle(prhs[1]));
    Mat6 P = mx_to_mat<6, 6>(prhs[2], "covariance");
    ekf->set_covariance(P);
}

static void mscekf_set_process_noise(int nlhs, mxArray* plhs[],
                                      int nrhs, const mxArray* prhs[]) {
    check_nargs(nrhs, 3, "mscekf_set_process_noise");
    auto* ekf = ekf_store().get(mx_to_handle(prhs[1]));
    double q = mx_to_scalar(prhs[2], "process_noise_intensity");
    ekf->motion_model().set_process_noise_intensity(q);
}

// ---------------------------------------------------------------------------
// GMPHD commands
// ---------------------------------------------------------------------------

static void gmphd_create(int nlhs, mxArray* plhs[],
                          int nrhs, const mxArray* prhs[]) {
    filters::GMPHD::Config cfg;
    if (nrhs > 1 && mxIsStruct(prhs[1])) {
        const mxArray* s = prhs[1];
        mxArray* f;
        if ((f = mxGetField(s, 0, "p_survival")))       cfg.p_survival = mxGetScalar(f);
        if ((f = mxGetField(s, 0, "p_detection")))       cfg.p_detection = mxGetScalar(f);
        if ((f = mxGetField(s, 0, "clutter_rate")))      cfg.clutter_rate = mxGetScalar(f);
        if ((f = mxGetField(s, 0, "merge_threshold")))   cfg.merge_threshold = mxGetScalar(f);
        if ((f = mxGetField(s, 0, "prune_threshold")))   cfg.prune_threshold = mxGetScalar(f);
        if ((f = mxGetField(s, 0, "max_components")))    cfg.max_components = static_cast<int>(mxGetScalar(f));
        if ((f = mxGetField(s, 0, "extraction_threshold"))) cfg.extraction_threshold = mxGetScalar(f);
    }
    uint64_t h = gmphd_store().create(cfg);
    plhs[0] = handle_to_mx(h);
}

static void gmphd_destroy(int nlhs, mxArray* plhs[],
                           int nrhs, const mxArray* prhs[]) {
    check_nargs(nrhs, 2, "gmphd_destroy");
    uint64_t h = mx_to_handle(prhs[1]);
    gmphd_store().destroy(h);
}

static void gmphd_predict(int nlhs, mxArray* plhs[],
                           int nrhs, const mxArray* prhs[]) {
    check_nargs(nrhs, 3, "gmphd_predict");
    auto* phd = gmphd_store().get(mx_to_handle(prhs[1]));
    double dt = mx_to_scalar(prhs[2], "dt");
    phd->predict(dt);
}

static void gmphd_correct(int nlhs, mxArray* plhs[],
                           int nrhs, const mxArray* prhs[]) {
    check_nargs(nrhs, 3, "gmphd_correct");
    auto* phd = gmphd_store().get(mx_to_handle(prhs[1]));
    auto measurements = mx_to_vecx_list(prhs[2], "measurements");
    phd->correct(measurements);
}

static void gmphd_merge(int nlhs, mxArray* plhs[],
                         int nrhs, const mxArray* prhs[]) {
    check_nargs(nrhs, 2, "gmphd_merge");
    auto* phd = gmphd_store().get(mx_to_handle(prhs[1]));
    phd->merge();
}

static void gmphd_prune(int nlhs, mxArray* plhs[],
                         int nrhs, const mxArray* prhs[]) {
    check_nargs(nrhs, 2, "gmphd_prune");
    auto* phd = gmphd_store().get(mx_to_handle(prhs[1]));
    phd->prune();
}

static void gmphd_extract(int nlhs, mxArray* plhs[],
                           int nrhs, const mxArray* prhs[]) {
    check_nargs(nrhs, 2, "gmphd_extract");
    auto* phd = gmphd_store().get(mx_to_handle(prhs[1]));
    auto targets = phd->extract();

    // Return struct array with fields: weight, mean, covariance
    const char* fields[] = {"weight", "mean", "covariance"};
    size_t n = targets.size();
    mxArray* out = mxCreateStructMatrix(n, 1, 3, fields);
    for (size_t i = 0; i < n; ++i) {
        mxSetField(out, i, "weight", scalar_to_mx(targets[i].weight));
        mxSetField(out, i, "mean", vecx_to_mx(targets[i].mean));
        mxSetField(out, i, "covariance", matxr_to_mx(targets[i].covariance));
    }
    plhs[0] = out;
}

static void gmphd_target_count(int nlhs, mxArray* plhs[],
                                int nrhs, const mxArray* prhs[]) {
    check_nargs(nrhs, 2, "gmphd_target_count");
    auto* phd = gmphd_store().get(mx_to_handle(prhs[1]));
    plhs[0] = scalar_to_mx(phd->estimated_target_count());
}

static void gmphd_add_birth(int nlhs, mxArray* plhs[],
                             int nrhs, const mxArray* prhs[]) {
    // args: handle, struct array with weight/mean/covariance
    check_nargs(nrhs, 3, "gmphd_add_birth");
    auto* phd = gmphd_store().get(mx_to_handle(prhs[1]));

    if (!mxIsStruct(prhs[2])) {
        mexErrMsgIdAndTxt("aot:type", "Birth components must be a struct array");
    }
    size_t n = mxGetNumberOfElements(prhs[2]);
    std::vector<filters::GaussianComponent> births(n);
    for (size_t i = 0; i < n; ++i) {
        mxArray* fw = mxGetField(prhs[2], i, "weight");
        mxArray* fm = mxGetField(prhs[2], i, "mean");
        mxArray* fp = mxGetField(prhs[2], i, "covariance");
        if (!fw || !fm || !fp) {
            mexErrMsgIdAndTxt("aot:field", "Birth components require weight, mean, covariance");
        }
        births[i].weight = mxGetScalar(fw);
        births[i].mean = mx_to_vecx(fm, "mean");
        births[i].covariance = mx_to_matxr(fp, "covariance");
    }
    phd->add_birth(births);
}

// ---------------------------------------------------------------------------
// Stateless functions: triangulation
// ---------------------------------------------------------------------------

static void triangulate_los_cmd(int nlhs, mxArray* plhs[],
                                 int nrhs, const mxArray* prhs[]) {
    // args: origins (Nx3), directions (Nx3), noises (cell array of 2x2 or single 2x2)
    check_nargs(nrhs, 3, "triangulate_los");

    const mxArray* origins_mx = prhs[1];
    const mxArray* dirs_mx = prhs[2];
    check_numeric(origins_mx, "origins");
    check_numeric(dirs_mx, "directions");

    size_t n = mxGetM(origins_mx);
    if (mxGetN(origins_mx) != 3 || mxGetM(dirs_mx) != n || mxGetN(dirs_mx) != 3) {
        mexErrMsgIdAndTxt("aot:size", "origins and directions must be Nx3");
    }

    const double* orig_data = mxGetPr(origins_mx);
    const double* dir_data = mxGetPr(dirs_mx);

    std::vector<LOSMeasurement> los(n);
    for (size_t i = 0; i < n; ++i) {
        los[i].origin(0) = orig_data[i];
        los[i].origin(1) = orig_data[n + i];
        los[i].origin(2) = orig_data[2 * n + i];
        los[i].direction(0) = dir_data[i];
        los[i].direction(1) = dir_data[n + i];
        los[i].direction(2) = dir_data[2 * n + i];
        los[i].noise = Mat2::Identity() * 1e-4;  // default noise
    }

    // Optional: noise argument
    if (nrhs > 3) {
        if (mxIsCell(prhs[3])) {
            for (size_t i = 0; i < n; ++i) {
                mxArray* cell = mxGetCell(prhs[3], i);
                if (cell) los[i].noise = mx_to_mat<2, 2>(cell, "noise");
            }
        } else if (mxIsDouble(prhs[3])) {
            Mat2 R = mx_to_mat<2, 2>(prhs[3], "noise");
            for (size_t i = 0; i < n; ++i) los[i].noise = R;
        }
    }

    auto result = fusion::triangulate_los(los);

    plhs[0] = vec_to_mx<3>(result.position);
    if (nlhs > 1) plhs[1] = mat_to_mx<3, 3>(result.covariance);
    if (nlhs > 2) plhs[2] = scalar_to_mx(result.residual);
    if (nlhs > 3) plhs[3] = bool_to_mx(result.valid);
}

// ---------------------------------------------------------------------------
// Stateless functions: assignment
// ---------------------------------------------------------------------------

static void gnn_assign_cmd(int nlhs, mxArray* plhs[],
                            int nrhs, const mxArray* prhs[]) {
    check_nargs(nrhs, 2, "gnn_assign");
    MatXR cost = mx_to_matxr(prhs[1], "cost_matrix");
    double gate = (nrhs > 2) ? mx_to_scalar(prhs[2], "gate_threshold") : 1e10;

    auto result = association::gnn_assign(cost, gate);

    // assignments: Mx2 matrix (1-indexed)
    size_t na = result.assignments.size();
    mxArray* assign = mxCreateDoubleMatrix(na, 2, mxREAL);
    double* ad = mxGetPr(assign);
    for (size_t i = 0; i < na; ++i) {
        ad[i] = static_cast<double>(result.assignments[i].first + 1);
        ad[na + i] = static_cast<double>(result.assignments[i].second + 1);
    }
    plhs[0] = assign;
    if (nlhs > 1) plhs[1] = int_vector_to_mx(result.unassigned_tracks);
    if (nlhs > 2) plhs[2] = int_vector_to_mx(result.unassigned_measurements);
    if (nlhs > 3) plhs[3] = scalar_to_mx(result.total_cost);
}

static void auction_assign_cmd(int nlhs, mxArray* plhs[],
                                int nrhs, const mxArray* prhs[]) {
    check_nargs(nrhs, 2, "auction_assign");
    MatXR cost = mx_to_matxr(prhs[1], "cost_matrix");
    double eps = (nrhs > 2) ? mx_to_scalar(prhs[2], "epsilon") : 1e-6;
    double gate = (nrhs > 3) ? mx_to_scalar(prhs[3], "gate_threshold") : 1e10;

    auto result = association::auction_assign(cost, eps, gate);

    size_t na = result.assignments.size();
    mxArray* assign = mxCreateDoubleMatrix(na, 2, mxREAL);
    double* ad = mxGetPr(assign);
    for (size_t i = 0; i < na; ++i) {
        ad[i] = static_cast<double>(result.assignments[i].first + 1);
        ad[na + i] = static_cast<double>(result.assignments[i].second + 1);
    }
    plhs[0] = assign;
    if (nlhs > 1) plhs[1] = int_vector_to_mx(result.unassigned_tracks);
    if (nlhs > 2) plhs[2] = int_vector_to_mx(result.unassigned_measurements);
    if (nlhs > 3) plhs[3] = scalar_to_mx(result.total_cost);
}

// ---------------------------------------------------------------------------
// Stateless functions: JPDA
// ---------------------------------------------------------------------------

static void jpda_probabilities_cmd(int nlhs, mxArray* plhs[],
                                    int nrhs, const mxArray* prhs[]) {
    check_nargs(nrhs, 2, "jpda_probabilities");
    MatXR likelihood = mx_to_matxr(prhs[1], "likelihood_matrix");
    double pd = (nrhs > 2) ? mx_to_scalar(prhs[2], "p_detection") : 0.9;
    double pg = (nrhs > 3) ? mx_to_scalar(prhs[3], "p_gate") : 0.99;
    double cd = (nrhs > 4) ? mx_to_scalar(prhs[4], "clutter_density") : 1e-6;

    auto result = association::jpda_probabilities(likelihood, pd, pg, cd);
    plhs[0] = matxr_to_mx(result.beta);
}

// ---------------------------------------------------------------------------
// Stateless functions: gating
// ---------------------------------------------------------------------------

static void mahalanobis_distance_cmd(int nlhs, mxArray* plhs[],
                                      int nrhs, const mxArray* prhs[]) {
    check_nargs(nrhs, 3, "mahalanobis_distance");
    VecX y = mx_to_vecx(prhs[1], "innovation");
    MatXR S = mx_to_matxr(prhs[2], "innovation_covariance");
    plhs[0] = scalar_to_mx(association::mahalanobis_distance(y, S));
}

static void gate_cmd(int nlhs, mxArray* plhs[],
                     int nrhs, const mxArray* prhs[]) {
    check_nargs(nrhs, 5, "gate");
    VecX predicted = mx_to_vecx(prhs[1], "predicted_measurement");
    auto measurements = mx_to_vecx_list(prhs[2], "measurements");
    MatXR S = mx_to_matxr(prhs[3], "innovation_covariance");
    double threshold = mx_to_scalar(prhs[4], "threshold");

    auto indices = association::gate(predicted, measurements, S, threshold);
    plhs[0] = int_vector_to_mx(indices);
}

static void chi2_gate_cmd(int nlhs, mxArray* plhs[],
                           int nrhs, const mxArray* prhs[]) {
    check_nargs(nrhs, 2, "chi2_gate");
    int dim = static_cast<int>(mx_to_scalar(prhs[1], "dimension"));
    double conf = (nrhs > 2) ? mx_to_scalar(prhs[2], "confidence") : 0.99;
    plhs[0] = scalar_to_mx(association::chi2_gate(dim, conf));
}

// ---------------------------------------------------------------------------
// Stateless functions: coordinate transforms
// ---------------------------------------------------------------------------

static void spherical_to_cartesian_cmd(int nlhs, mxArray* plhs[],
                                        int nrhs, const mxArray* prhs[]) {
    check_nargs(nrhs, 4, "spherical_to_cartesian");
    double az = mx_to_scalar(prhs[1], "az");
    double el = mx_to_scalar(prhs[2], "el");
    double r = mx_to_scalar(prhs[3], "r");
    plhs[0] = vec_to_mx<3>(coords::spherical_to_cartesian(az, el, r));
}

static void cartesian_to_spherical_cmd(int nlhs, mxArray* plhs[],
                                        int nrhs, const mxArray* prhs[]) {
    check_nargs(nrhs, 2, "cartesian_to_spherical");
    Vec3 pos = mx_to_vec<3>(prhs[1], "position");
    auto s = coords::cartesian_to_spherical(pos);
    plhs[0] = scalar_to_mx(s.az);
    if (nlhs > 1) plhs[1] = scalar_to_mx(s.el);
    if (nlhs > 2) plhs[2] = scalar_to_mx(s.r);
}

static void msc_to_cartesian_cmd(int nlhs, mxArray* plhs[],
                                  int nrhs, const mxArray* prhs[]) {
    check_nargs(nrhs, 4, "msc_to_cartesian");
    double az = mx_to_scalar(prhs[1], "az");
    double el = mx_to_scalar(prhs[2], "el");
    double inv_range = mx_to_scalar(prhs[3], "inv_range");
    plhs[0] = vec_to_mx<3>(coords::msc_to_cartesian(az, el, inv_range));
}

static void cartesian_to_msc_cmd(int nlhs, mxArray* plhs[],
                                  int nrhs, const mxArray* prhs[]) {
    check_nargs(nrhs, 2, "cartesian_to_msc");
    Vec3 pos = mx_to_vec<3>(prhs[1], "position");
    auto m = coords::cartesian_to_msc(pos);
    plhs[0] = scalar_to_mx(m.az);
    if (nlhs > 1) plhs[1] = scalar_to_mx(m.el);
    if (nlhs > 2) plhs[2] = scalar_to_mx(m.inv_range);
}

static void wrap_to_pi_cmd(int nlhs, mxArray* plhs[],
                            int nrhs, const mxArray* prhs[]) {
    check_nargs(nrhs, 2, "wrap_to_pi");
    double angle = mx_to_scalar(prhs[1], "angle");
    plhs[0] = scalar_to_mx(coords::wrap_to_pi(angle));
}

static void az_el_to_unit_vector_cmd(int nlhs, mxArray* plhs[],
                                      int nrhs, const mxArray* prhs[]) {
    check_nargs(nrhs, 3, "az_el_to_unit_vector");
    double az = mx_to_scalar(prhs[1], "az");
    double el = mx_to_scalar(prhs[2], "el");
    plhs[0] = vec_to_mx<3>(coords::az_el_to_unit_vector(az, el));
}

// ---------------------------------------------------------------------------
// Command dispatch table
// ---------------------------------------------------------------------------

using CmdFn = void(*)(int, mxArray*[], int, const mxArray*[]);

static const std::unordered_map<std::string, CmdFn>& command_table() {
    static const std::unordered_map<std::string, CmdFn> table = {
        // MSCEKF
        {"mscekf_create",               mscekf_create},
        {"mscekf_create_from_detection", mscekf_create_from_detection},
        {"mscekf_destroy",              mscekf_destroy},
        {"mscekf_predict",              mscekf_predict},
        {"mscekf_correct",              mscekf_correct},
        {"mscekf_correct_jpda",         mscekf_correct_jpda},
        {"mscekf_state",                mscekf_state},
        {"mscekf_covariance",           mscekf_covariance},
        {"mscekf_distance",             mscekf_distance},
        {"mscekf_likelihood",           mscekf_likelihood},
        {"mscekf_smooth",               mscekf_smooth},
        {"mscekf_set_store_history",    mscekf_set_store_history},
        {"mscekf_set_state",            mscekf_set_state},
        {"mscekf_set_covariance",       mscekf_set_covariance},
        {"mscekf_set_process_noise",    mscekf_set_process_noise},
        // GMPHD
        {"gmphd_create",                gmphd_create},
        {"gmphd_destroy",               gmphd_destroy},
        {"gmphd_predict",               gmphd_predict},
        {"gmphd_correct",               gmphd_correct},
        {"gmphd_merge",                 gmphd_merge},
        {"gmphd_prune",                 gmphd_prune},
        {"gmphd_extract",               gmphd_extract},
        {"gmphd_target_count",          gmphd_target_count},
        {"gmphd_add_birth",             gmphd_add_birth},
        // Triangulation
        {"triangulate_los",             triangulate_los_cmd},
        // Assignment
        {"gnn_assign",                  gnn_assign_cmd},
        {"auction_assign",              auction_assign_cmd},
        // JPDA
        {"jpda_probabilities",          jpda_probabilities_cmd},
        // Gating
        {"mahalanobis_distance",        mahalanobis_distance_cmd},
        {"gate",                        gate_cmd},
        {"chi2_gate",                   chi2_gate_cmd},
        // Coordinate transforms
        {"spherical_to_cartesian",      spherical_to_cartesian_cmd},
        {"cartesian_to_spherical",      cartesian_to_spherical_cmd},
        {"msc_to_cartesian",            msc_to_cartesian_cmd},
        {"cartesian_to_msc",            cartesian_to_msc_cmd},
        {"wrap_to_pi",                  wrap_to_pi_cmd},
        {"az_el_to_unit_vector",        az_el_to_unit_vector_cmd},
    };
    return table;
}

// ---------------------------------------------------------------------------
// MEX entry point
// ---------------------------------------------------------------------------

void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[]) {
    // Register cleanup on first call
    static bool registered = false;
    if (!registered) {
        mexAtExit(cleanup);
        registered = true;
    }

    if (nrhs < 1) {
        mexErrMsgIdAndTxt("aot:nargs",
            "Usage: angle_only_mex('command', ...)");
    }

    std::string cmd = mx_to_string(prhs[0]);

    auto& table = command_table();
    auto it = table.find(cmd);
    if (it == table.end()) {
        mexErrMsgIdAndTxt("aot:command", "Unknown command: %s", cmd.c_str());
    }

    it->second(nlhs, plhs, nrhs, prhs);
}
