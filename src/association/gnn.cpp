#include "angle_only/association/gnn.hpp"
#include <algorithm>
#include <limits>
#include <numeric>
#include <cmath>

namespace aot::association {

namespace {

// Hungarian algorithm (Kuhn-Munkres) for minimum cost assignment
class Hungarian {
public:
    explicit Hungarian(const MatXR& cost, double gate)
        : n_(std::max(cost.rows(), cost.cols()))
        , cost_(n_, n_)
        , orig_rows_(cost.rows())
        , orig_cols_(cost.cols())
        , gate_(gate)
    {
        cost_.setConstant(gate);
        cost_.topLeftCorner(cost.rows(), cost.cols()) = cost;
    }

    AssignmentResult solve() {
        auto n = static_cast<int>(n_);
        std::vector<double> u(static_cast<size_t>(n) + 1, 0);
        std::vector<double> v(static_cast<size_t>(n) + 1, 0);
        std::vector<int> p(static_cast<size_t>(n) + 1, 0);
        std::vector<int> way(static_cast<size_t>(n) + 1, 0);

        for (int i = 1; i <= n; ++i) {
            p[0] = i;
            int j0 = 0;
            std::vector<double> minv(static_cast<size_t>(n) + 1, std::numeric_limits<double>::max());
            std::vector<bool> used(static_cast<size_t>(n) + 1, false);

            do {
                used[static_cast<size_t>(j0)] = true;
                int i0 = p[static_cast<size_t>(j0)];
                double delta = std::numeric_limits<double>::max();
                int j1 = -1;

                for (int j = 1; j <= n; ++j) {
                    if (used[static_cast<size_t>(j)]) continue;
                    double cur = cost_(i0 - 1, j - 1) - u[static_cast<size_t>(i0)] - v[static_cast<size_t>(j)];
                    if (cur < minv[static_cast<size_t>(j)]) {
                        minv[static_cast<size_t>(j)] = cur;
                        way[static_cast<size_t>(j)] = j0;
                    }
                    if (minv[static_cast<size_t>(j)] < delta) {
                        delta = minv[static_cast<size_t>(j)];
                        j1 = j;
                    }
                }

                for (int j = 0; j <= n; ++j) {
                    if (used[static_cast<size_t>(j)]) {
                        u[static_cast<size_t>(p[static_cast<size_t>(j)])] += delta;
                        v[static_cast<size_t>(j)] -= delta;
                    } else {
                        minv[static_cast<size_t>(j)] -= delta;
                    }
                }

                j0 = j1;
            } while (p[static_cast<size_t>(j0)] != 0);

            do {
                int j1 = way[static_cast<size_t>(j0)];
                p[static_cast<size_t>(j0)] = p[static_cast<size_t>(j1)];
                j0 = j1;
            } while (j0 != 0);
        }

        // Extract results
        AssignmentResult result;
        std::vector<bool> track_assigned(static_cast<size_t>(orig_rows_), false);
        std::vector<bool> meas_assigned(static_cast<size_t>(orig_cols_), false);

        for (int j = 1; j <= n; ++j) {
            int i = p[static_cast<size_t>(j)] - 1;
            int jj = j - 1;
            if (i < orig_rows_ && jj < orig_cols_ && cost_(i, jj) < gate_) {
                result.assignments.emplace_back(i, jj);
                track_assigned[static_cast<size_t>(i)] = true;
                meas_assigned[static_cast<size_t>(jj)] = true;
                result.total_cost += cost_(i, jj);
            }
        }

        for (int i = 0; i < orig_rows_; ++i) {
            if (!track_assigned[static_cast<size_t>(i)])
                result.unassigned_tracks.push_back(i);
        }
        for (int j = 0; j < orig_cols_; ++j) {
            if (!meas_assigned[static_cast<size_t>(j)])
                result.unassigned_measurements.push_back(j);
        }

        return result;
    }

private:
    Eigen::Index n_;
    MatXR cost_;
    Eigen::Index orig_rows_, orig_cols_;
    double gate_;
};

} // anonymous namespace

AssignmentResult gnn_assign(const MatXR& cost_matrix, double gate_threshold) {
    if (cost_matrix.rows() == 0 || cost_matrix.cols() == 0) {
        AssignmentResult result;
        for (int i = 0; i < cost_matrix.rows(); ++i)
            result.unassigned_tracks.push_back(i);
        for (int j = 0; j < cost_matrix.cols(); ++j)
            result.unassigned_measurements.push_back(j);
        return result;
    }
    Hungarian h(cost_matrix, gate_threshold);
    return h.solve();
}

AssignmentResult auction_assign(const MatXR& cost_matrix, double epsilon, double gate_threshold) {
    // Simplified auction algorithm
    auto n_tracks = static_cast<int>(cost_matrix.rows());
    auto n_meas = static_cast<int>(cost_matrix.cols());

    std::vector<int> assignment(static_cast<size_t>(n_tracks), -1);
    std::vector<int> meas_owner(static_cast<size_t>(n_meas), -1);
    std::vector<double> prices(static_cast<size_t>(n_meas), 0.0);

    int max_iter = n_tracks * n_meas + 1;
    for (int iter = 0; iter < max_iter; ++iter) {
        bool all_assigned = true;

        for (int i = 0; i < n_tracks; ++i) {
            if (assignment[static_cast<size_t>(i)] >= 0) continue;
            all_assigned = false;

            // Find best and second best value
            double best_val = -std::numeric_limits<double>::max();
            double second_val = -std::numeric_limits<double>::max();
            int best_j = -1;

            for (int j = 0; j < n_meas; ++j) {
                double val = -(cost_matrix(i, j) + prices[static_cast<size_t>(j)]);
                if (val > best_val) {
                    second_val = best_val;
                    best_val = val;
                    best_j = j;
                } else if (val > second_val) {
                    second_val = val;
                }
            }

            if (best_j < 0) continue;
            if (cost_matrix(i, best_j) > gate_threshold) continue;

            // Bid
            double bid = best_val - second_val + epsilon;
            prices[static_cast<size_t>(best_j)] += bid;

            // Evict previous owner
            int prev = meas_owner[static_cast<size_t>(best_j)];
            if (prev >= 0) {
                assignment[static_cast<size_t>(prev)] = -1;
            }

            assignment[static_cast<size_t>(i)] = best_j;
            meas_owner[static_cast<size_t>(best_j)] = i;
        }

        if (all_assigned) break;
    }

    AssignmentResult result;
    std::vector<bool> meas_used(static_cast<size_t>(n_meas), false);
    for (int i = 0; i < n_tracks; ++i) {
        int j = assignment[static_cast<size_t>(i)];
        if (j >= 0 && cost_matrix(i, j) <= gate_threshold) {
            result.assignments.emplace_back(i, j);
            meas_used[static_cast<size_t>(j)] = true;
            result.total_cost += cost_matrix(i, j);
        } else {
            result.unassigned_tracks.push_back(i);
        }
    }
    for (int j = 0; j < n_meas; ++j) {
        if (!meas_used[static_cast<size_t>(j)])
            result.unassigned_measurements.push_back(j);
    }
    return result;
}

} // namespace aot::association
