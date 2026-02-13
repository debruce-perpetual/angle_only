#pragma once

#include "angle_only/core/types.hpp"
#include <vector>
#include <functional>

namespace aot::filters {

/// A single Gaussian component in the GM-PHD mixture
struct GaussianComponent {
    double weight = 0.0;
    VecX mean;
    MatXR covariance;

    GaussianComponent() = default;
    GaussianComponent(double w, VecX m, MatXR P)
        : weight(w), mean(std::move(m)), covariance(std::move(P)) {}
};

/// Gaussian Mixture Probability Hypothesis Density (GM-PHD) filter.
/// Multi-target tracker that estimates the intensity function (PHD)
/// as a Gaussian mixture. Handles unknown and time-varying number of targets.
class GMPHD {
public:
    struct Config {
        double p_survival = 0.99;
        double p_detection = 0.9;
        double clutter_rate = 1e-5;
        double merge_threshold = 4.0;
        double prune_threshold = 1e-5;
        int max_components = 100;
        double extraction_threshold = 0.5;
        Config() = default;
    };

    using TransitionFn = std::function<VecX(const VecX&, double)>;
    using TransitionJacFn = std::function<MatXR(const VecX&, double)>;
    using ProcessNoiseFn = std::function<MatXR(double)>;
    using MeasurementFn = std::function<VecX(const VecX&)>;
    using MeasurementJacFn = std::function<MatXR(const VecX&)>;

    GMPHD();
    explicit GMPHD(Config config);

    // Set dynamics models
    void set_transition(TransitionFn f, TransitionJacFn jac, ProcessNoiseFn Q);
    void set_measurement(MeasurementFn h, MeasurementJacFn jac, MatXR R);

    /// Add birth components (new potential targets)
    void add_birth(const std::vector<GaussianComponent>& births);

    /// Range-parameterized birth from angular detection
    void add_birth_from_detection(const VecX& measurement, const MatXR& R_meas,
                                   const std::vector<double>& range_hypotheses);

    /// Predict step
    void predict(double dt);

    /// Correct/update step with measurements
    void correct(const std::vector<VecX>& measurements);

    /// Merge close components
    void merge();

    /// Prune low-weight components
    void prune();

    /// Extract target state estimates (components with weight > threshold)
    [[nodiscard]] std::vector<GaussianComponent> extract() const;

    /// Get all current components
    [[nodiscard]] const std::vector<GaussianComponent>& components() const { return components_; }

    /// Get estimated number of targets
    [[nodiscard]] double estimated_target_count() const;

    [[nodiscard]] const Config& config() const { return config_; }
    void set_config(const Config& c) { config_ = c; }

private:
    Config config_;
    std::vector<GaussianComponent> components_;
    std::vector<GaussianComponent> birth_components_;

    TransitionFn f_;
    TransitionJacFn f_jac_;
    ProcessNoiseFn Q_fn_;
    MeasurementFn h_;
    MeasurementJacFn h_jac_;
    MatXR R_;
};

} // namespace aot::filters
