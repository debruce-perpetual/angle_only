#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "angle_only/filters/gmphd.hpp"
#include <cmath>

using namespace aot;
using namespace aot::filters;
using Catch::Matchers::WithinAbs;

TEST_CASE("GMPHD birth and predict", "[filters][gmphd]") {
    GMPHD::Config config;
    config.p_survival = 0.99;
    config.p_detection = 0.9;
    GMPHD phd(config);

    // Set simple linear dynamics
    phd.set_transition(
        [](const VecX& x, double dt) -> VecX {
            VecX xp = x;
            xp.head(3) += x.tail(3) * dt;
            return xp;
        },
        [](const VecX& /*x*/, double dt) -> MatXR {
            MatXR F = MatXR::Identity(6, 6);
            F(0, 3) = dt; F(1, 4) = dt; F(2, 5) = dt;
            return F;
        },
        [](double dt) -> MatXR {
            return MatXR::Identity(6, 6) * dt * 0.01;
        }
    );

    // Add birth component
    VecX birth_mean = VecX::Zero(6);
    birth_mean(0) = 0.5;
    MatXR birth_cov = MatXR::Identity(6, 6);
    phd.add_birth({{1.0, birth_mean, birth_cov}});

    phd.predict(1.0);

    REQUIRE(phd.components().size() == 1);
    REQUIRE_THAT(phd.components()[0].weight, WithinAbs(0.99, 1e-10));
}

TEST_CASE("GMPHD correct increases components", "[filters][gmphd]") {
    GMPHD::Config config;
    config.p_detection = 0.9;
    config.clutter_rate = 1e-5;
    GMPHD phd(config);

    phd.set_transition(
        [](const VecX& x, double) -> VecX { return x; },
        [](const VecX&, double) -> MatXR { return MatXR::Identity(6, 6); },
        [](double dt) -> MatXR { return MatXR::Identity(6, 6) * dt * 0.01; }
    );

    MatXR R = MatXR::Identity(2, 2) * 1e-4;
    phd.set_measurement(
        [](const VecX& x) -> VecX { return x.head(2); },
        [](const VecX&) -> MatXR {
            MatXR H = MatXR::Zero(2, 6);
            H(0, 0) = 1.0; H(1, 1) = 1.0;
            return H;
        },
        R
    );

    // Add birth
    VecX birth = VecX::Zero(6);
    birth(0) = 0.5; birth(1) = 0.3;
    phd.add_birth({{1.0, birth, MatXR::Identity(6, 6)}});

    phd.predict(1.0);
    size_t before = phd.components().size();

    // Correct with one measurement
    VecX z(2);
    z << 0.5, 0.3;
    phd.correct({z});

    // Should have missed-detection + one detection update
    REQUIRE(phd.components().size() > before);
}

TEST_CASE("GMPHD merge reduces components", "[filters][gmphd]") {
    GMPHD phd;

    // Manually add close components
    VecX m1 = VecX::Zero(6); m1(0) = 0.5;
    VecX m2 = VecX::Zero(6); m2(0) = 0.5001;
    MatXR P = MatXR::Identity(6, 6);

    std::vector<GaussianComponent> births;
    births.emplace_back(0.5, m1, P);
    births.emplace_back(0.3, m2, P);
    phd.add_birth(births);

    // Need to predict to move births into components
    phd.set_transition(
        [](const VecX& x, double) -> VecX { return x; },
        [](const VecX&, double) -> MatXR { return MatXR::Identity(6, 6); },
        [](double) -> MatXR { return MatXR::Identity(6, 6) * 0.01; }
    );
    phd.predict(0.0);

    size_t before = phd.components().size();
    phd.merge();
    REQUIRE(phd.components().size() <= before);
}

TEST_CASE("GMPHD prune removes low weight", "[filters][gmphd]") {
    GMPHD::Config config;
    config.prune_threshold = 0.1;
    GMPHD phd(config);

    phd.set_transition(
        [](const VecX& x, double) -> VecX { return x; },
        [](const VecX&, double) -> MatXR { return MatXR::Identity(6, 6); },
        [](double) -> MatXR { return MatXR::Identity(6, 6) * 0.01; }
    );

    VecX m = VecX::Zero(6);
    MatXR P = MatXR::Identity(6, 6);
    phd.add_birth({{0.5, m, P}, {0.01, m, P}});
    phd.predict(0.0);

    phd.prune();
    REQUIRE(phd.components().size() == 1);
}

TEST_CASE("GMPHD estimated target count", "[filters][gmphd]") {
    GMPHD phd;

    phd.set_transition(
        [](const VecX& x, double) -> VecX { return x; },
        [](const VecX&, double) -> MatXR { return MatXR::Identity(6, 6); },
        [](double) -> MatXR { return MatXR::Identity(6, 6) * 0.01; }
    );

    VecX m = VecX::Zero(6);
    MatXR P = MatXR::Identity(6, 6);
    phd.add_birth({{1.0, m, P}, {1.0, m, P}});
    phd.predict(0.0);

    REQUIRE_THAT(phd.estimated_target_count(), WithinAbs(2.0 * 0.99, 0.1));
}

TEST_CASE("GMPHD range-parameterized birth", "[filters][gmphd]") {
    GMPHD phd;

    phd.set_transition(
        [](const VecX& x, double) -> VecX { return x; },
        [](const VecX&, double) -> MatXR { return MatXR::Identity(6, 6); },
        [](double) -> MatXR { return MatXR::Identity(6, 6) * 0.01; }
    );

    VecX z(2);
    z << 0.5, 0.3;
    MatXR R = MatXR::Identity(2, 2) * 1e-4;

    phd.add_birth_from_detection(z, R, {100.0, 500.0, 1000.0});
    phd.predict(0.0);

    REQUIRE(phd.components().size() == 3);
}
