#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "angle_only/association/gating.hpp"
#include "angle_only/association/gnn.hpp"
#include "angle_only/association/jpda.hpp"

using namespace aot;
using namespace aot::association;
using Catch::Matchers::WithinAbs;

TEST_CASE("Mahalanobis distance identity covariance", "[association][gating]") {
    VecX y(2);
    y << 3.0, 4.0;
    MatXR S = MatXR::Identity(2, 2);

    double d = mahalanobis_distance(y, S);
    REQUIRE_THAT(d, WithinAbs(5.0, 1e-10));
}

TEST_CASE("Gating filters measurements", "[association][gating]") {
    VecX pred(2);
    pred << 0.0, 0.0;

    std::vector<VecX> meas;
    for (int i = 0; i < 5; ++i) {
        VecX z(2);
        z << 0.1 * i, 0.0;
        meas.push_back(z);
    }

    MatXR S = MatXR::Identity(2, 2);
    auto gated = gate(pred, meas, S, 0.25);  // threshold = 0.25

    // Should include measurements 0 (d=0), 1 (d=0.1), 2 (d=0.2)
    REQUIRE(gated.size() == 3);
}

TEST_CASE("Chi2 gate values", "[association][gating]") {
    double g = chi2_gate(2, 0.99);
    REQUIRE_THAT(g, WithinAbs(9.210, 0.001));

    g = chi2_gate(2, 0.95);
    REQUIRE_THAT(g, WithinAbs(5.991, 0.001));
}

TEST_CASE("GNN assignment simple case", "[association][gnn]") {
    MatXR cost(3, 3);
    cost << 1, 10, 10,
            10, 2, 10,
            10, 10, 3;

    auto result = gnn_assign(cost);
    REQUIRE(result.assignments.size() == 3);
    REQUIRE(result.unassigned_tracks.empty());
    REQUIRE(result.unassigned_measurements.empty());
    REQUIRE_THAT(result.total_cost, WithinAbs(6.0, 1e-10));
}

TEST_CASE("GNN assignment with gating", "[association][gnn]") {
    MatXR cost(2, 3);
    cost << 1, 100, 100,
            100, 2, 100;

    auto result = gnn_assign(cost, 50.0);
    REQUIRE(result.assignments.size() == 2);
    REQUIRE(result.unassigned_measurements.size() == 1);
}

TEST_CASE("GNN assignment empty matrix", "[association][gnn]") {
    MatXR cost(0, 0);
    auto result = gnn_assign(cost);
    REQUIRE(result.assignments.empty());
}

TEST_CASE("Auction assignment matches GNN", "[association][gnn]") {
    MatXR cost(3, 3);
    cost << 1, 10, 10,
            10, 2, 10,
            10, 10, 3;

    auto gnn_result = gnn_assign(cost);
    auto auction_result = auction_assign(cost);

    REQUIRE_THAT(auction_result.total_cost, WithinAbs(gnn_result.total_cost, 1.0));
}

TEST_CASE("JPDA probabilities sum to 1", "[association][jpda]") {
    MatXR likelihood(2, 3);
    likelihood << 0.8, 0.1, 0.01,
                  0.1, 0.7, 0.02;

    auto result = jpda_probabilities(likelihood);

    for (Eigen::Index i = 0; i < result.beta.rows(); ++i) {
        double sum = result.beta.row(i).sum();
        REQUIRE_THAT(sum, WithinAbs(1.0, 1e-10));
    }
}
