#include "angle_only/filters/gmphd.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace aot::filters {

GMPHD::GMPHD() : config_() {}
GMPHD::GMPHD(Config config) : config_(std::move(config)) {}

void GMPHD::set_transition(TransitionFn f, TransitionJacFn jac, ProcessNoiseFn Q) {
    f_ = std::move(f);
    f_jac_ = std::move(jac);
    Q_fn_ = std::move(Q);
}

void GMPHD::set_measurement(MeasurementFn h, MeasurementJacFn jac, MatXR R) {
    h_ = std::move(h);
    h_jac_ = std::move(jac);
    R_ = std::move(R);
}

void GMPHD::add_birth(const std::vector<GaussianComponent>& births) {
    birth_components_.insert(birth_components_.end(), births.begin(), births.end());
}

void GMPHD::add_birth_from_detection(const VecX& measurement, const MatXR& R_meas,
                                       const std::vector<double>& range_hypotheses) {
    double w_each = 0.1 / static_cast<double>(range_hypotheses.size());
    auto meas_dim = measurement.size();

    for (double r : range_hypotheses) {
        // Create state from measurement + range hypothesis
        // State: [az, el, 1/r, az_rate, el_rate, inv_range_rate]
        VecX state = VecX::Zero(6);
        if (meas_dim >= 2) {
            state(0) = measurement(0);  // az
            state(1) = measurement(1);  // el
        }
        state(2) = 1.0 / r;  // inverse range

        MatXR P = MatXR::Zero(6, 6);
        if (meas_dim >= 2) {
            P(0, 0) = R_meas(0, 0);
            P(1, 1) = R_meas(1, 1);
        }
        double inv_r_std = 0.3 / r;  // 30% range uncertainty
        P(2, 2) = inv_r_std * inv_r_std;
        P(3, 3) = 1.0;   // rate uncertainties
        P(4, 4) = 1.0;
        P(5, 5) = 0.01;

        birth_components_.emplace_back(w_each, std::move(state), std::move(P));
    }
}

void GMPHD::predict(double dt) {
    // Add birth components
    components_.insert(components_.end(),
                       birth_components_.begin(), birth_components_.end());
    birth_components_.clear();

    // Predict each component
    for (auto& comp : components_) {
        comp.weight *= config_.p_survival;

        MatXR F = f_jac_(comp.mean, dt);
        MatXR Q = Q_fn_(dt);
        comp.mean = f_(comp.mean, dt);
        comp.covariance = F * comp.covariance * F.transpose() + Q;
    }
}

void GMPHD::correct(const std::vector<VecX>& measurements) {
    if (!h_ || !h_jac_) return;

    // Save predicted components (will become missed-detection terms)
    std::vector<GaussianComponent> updated;
    updated.reserve(components_.size() * (measurements.size() + 1));

    // Missed detection components (weight scaled by 1 - p_d)
    for (const auto& comp : components_) {
        updated.emplace_back(
            comp.weight * (1.0 - config_.p_detection),
            comp.mean, comp.covariance);
    }

    // Detection update for each measurement
    for (const auto& z : measurements) {
        double likelihood_sum = 0.0;
        std::vector<GaussianComponent> z_updated;
        z_updated.reserve(components_.size());

        for (const auto& comp : components_) {
            MatXR H = h_jac_(comp.mean);
            VecX z_pred = h_(comp.mean);
            VecX y = z - z_pred;

            MatXR S = H * comp.covariance * H.transpose() + R_;
            MatXR K = comp.covariance * H.transpose() * S.inverse();

            VecX mean_upd = comp.mean + K * y;
            auto state_dim = comp.mean.size();
            MatXR P_upd = (MatXR::Identity(state_dim, state_dim) - K * H) * comp.covariance;

            // Gaussian likelihood
            double det_S = S.determinant();
            auto meas_dim = z.size();
            double mah2 = static_cast<double>(y.transpose() * S.inverse() * y);
            double log_2pi = 1.8378770664093453;
            double lik = std::exp(-0.5 * (mah2 + static_cast<double>(meas_dim) * log_2pi + std::log(std::abs(det_S) + 1e-300)));

            double w_upd = config_.p_detection * comp.weight * lik;
            likelihood_sum += w_upd;

            z_updated.emplace_back(w_upd, std::move(mean_upd), std::move(P_upd));
        }

        // Normalize by clutter + sum of likelihoods
        double normalizer = config_.clutter_rate + likelihood_sum;
        for (auto& comp : z_updated) {
            comp.weight /= normalizer;
        }

        updated.insert(updated.end(), z_updated.begin(), z_updated.end());
    }

    components_ = std::move(updated);
}

void GMPHD::merge() {
    if (components_.empty()) return;

    std::vector<bool> merged(components_.size(), false);
    std::vector<GaussianComponent> result;

    // Sort by weight descending
    std::vector<size_t> indices(components_.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return components_[a].weight > components_[b].weight;
    });

    for (size_t idx : indices) {
        if (merged[idx]) continue;

        auto& seed = components_[idx];
        double w_sum = seed.weight;
        VecX mean_sum = seed.weight * seed.mean;
        MatXR cov_sum = seed.weight * (seed.covariance + seed.mean * seed.mean.transpose());
        merged[idx] = true;

        for (size_t j = 0; j < components_.size(); ++j) {
            if (merged[j]) continue;

            auto& other = components_[j];
            VecX diff = other.mean - seed.mean;

            // Mahalanobis distance
            double d2 = static_cast<double>(diff.transpose() * seed.covariance.inverse() * diff);
            if (d2 <= config_.merge_threshold) {
                w_sum += other.weight;
                mean_sum += other.weight * other.mean;
                cov_sum += other.weight * (other.covariance + other.mean * other.mean.transpose());
                merged[j] = true;
            }
        }

        VecX new_mean = mean_sum / w_sum;
        MatXR new_cov = cov_sum / w_sum - new_mean * new_mean.transpose();

        result.emplace_back(w_sum, std::move(new_mean), std::move(new_cov));
    }

    components_ = std::move(result);
}

void GMPHD::prune() {
    auto it = std::remove_if(components_.begin(), components_.end(),
        [this](const GaussianComponent& c) {
            return c.weight < config_.prune_threshold;
        });
    components_.erase(it, components_.end());

    // Cap at max components (keep highest weight)
    if (static_cast<int>(components_.size()) > config_.max_components) {
        std::sort(components_.begin(), components_.end(),
            [](const GaussianComponent& a, const GaussianComponent& b) {
                return a.weight > b.weight;
            });
        components_.resize(static_cast<size_t>(config_.max_components));
    }
}

std::vector<GaussianComponent> GMPHD::extract() const {
    std::vector<GaussianComponent> targets;
    for (const auto& comp : components_) {
        if (comp.weight >= config_.extraction_threshold) {
            targets.push_back(comp);
        }
    }
    return targets;
}

double GMPHD::estimated_target_count() const {
    double sum = 0.0;
    for (const auto& comp : components_) {
        sum += comp.weight;
    }
    return sum;
}

} // namespace aot::filters
