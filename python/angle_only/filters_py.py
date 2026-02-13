"""Pythonic wrappers for tracking filters."""

from __future__ import annotations

import numpy as np
from typing import Optional, Sequence

try:
    from angle_only._angle_only_cpp import core, filters as _filters
except ImportError:
    _filters = None
    core = None


class MSCEKF:
    """Modified Spherical Coordinates Extended Kalman Filter.

    Wraps the C++ aot::filters::MSCEKF with a Pythonic interface.

    Example::

        ekf = MSCEKF.from_detection(azimuth=0.5, elevation=0.3)
        ekf.predict(dt=1.0)
        ekf.correct(measurement=np.array([0.51, 0.29]),
                     noise=np.eye(2) * 1e-4)
    """

    def __init__(self, cpp_ekf=None):
        if cpp_ekf is not None:
            self._ekf = cpp_ekf
        else:
            self._ekf = _filters.MSCEKF()

    @classmethod
    def from_detection(cls, azimuth: float = 0.0, elevation: float = 0.0,
                       noise: Optional[np.ndarray] = None,
                       initial_inv_range: float = 0.01,
                       inv_range_std: float = 0.05) -> "MSCEKF":
        """Initialize filter from a single angular detection."""
        det = core.Detection()
        det.azimuth = azimuth
        det.elevation = elevation
        if noise is not None:
            det.noise = np.asarray(noise, dtype=np.float64)
        else:
            det.noise = np.eye(2) * 1e-4
        cpp_ekf = _filters.MSCEKF.from_detection(det, initial_inv_range, inv_range_std)
        return cls(cpp_ekf)

    def predict(self, dt: float) -> None:
        """Predict state forward by dt seconds."""
        self._ekf.predict(dt)

    def correct(self, measurement: np.ndarray,
                noise: Optional[np.ndarray] = None) -> None:
        """Correct state with angular measurement [az, el]."""
        z = np.asarray(measurement, dtype=np.float64).ravel()[:2]
        R = np.asarray(noise, dtype=np.float64) if noise is not None else np.eye(2) * 1e-4
        self._ekf.correct(z, R)

    def correct_jpda(self, measurements: Sequence[np.ndarray],
                     weights: Sequence[float],
                     noise: Optional[np.ndarray] = None) -> None:
        """JPDA correction with weighted measurements."""
        meas = [np.asarray(m, dtype=np.float64).ravel()[:2] for m in measurements]
        R = np.asarray(noise, dtype=np.float64) if noise is not None else np.eye(2) * 1e-4
        self._ekf.correct_jpda(meas, list(weights), R)

    def distance(self, measurement: np.ndarray,
                 noise: Optional[np.ndarray] = None) -> float:
        """Mahalanobis distance to measurement."""
        z = np.asarray(measurement, dtype=np.float64).ravel()[:2]
        R = np.asarray(noise, dtype=np.float64) if noise is not None else np.eye(2) * 1e-4
        return self._ekf.distance(z, R)

    def likelihood(self, measurement: np.ndarray,
                   noise: Optional[np.ndarray] = None) -> float:
        """Measurement likelihood (Gaussian)."""
        z = np.asarray(measurement, dtype=np.float64).ravel()[:2]
        R = np.asarray(noise, dtype=np.float64) if noise is not None else np.eye(2) * 1e-4
        return self._ekf.likelihood(z, R)

    def smooth(self) -> list[np.ndarray]:
        """Run RTS fixed-interval smoother on stored history."""
        return self._ekf.smooth()

    @property
    def state(self) -> np.ndarray:
        """Current state vector [az, el, 1/r, az_rate, el_rate, inv_range_rate]."""
        return np.array(self._ekf.state)

    @state.setter
    def state(self, x: np.ndarray) -> None:
        self._ekf.state = np.asarray(x, dtype=np.float64)

    @property
    def covariance(self) -> np.ndarray:
        """Current state covariance (6x6)."""
        return np.array(self._ekf.covariance)

    @covariance.setter
    def covariance(self, P: np.ndarray) -> None:
        self._ekf.covariance = np.asarray(P, dtype=np.float64)

    @property
    def azimuth(self) -> float:
        return float(self._ekf.state[0])

    @property
    def elevation(self) -> float:
        return float(self._ekf.state[1])

    @property
    def inv_range(self) -> float:
        return float(self._ekf.state[2])

    def set_store_history(self, store: bool) -> None:
        self._ekf.set_store_history(store)

    def clear_history(self) -> None:
        self._ekf.clear_history()

    def __repr__(self) -> str:
        s = self._ekf.state
        return f"MSCEKF(az={s[0]:.4f}, el={s[1]:.4f}, inv_r={s[2]:.6f})"


class GMPHD:
    """Gaussian Mixture PHD filter for multi-target tracking.

    Example::

        phd = GMPHD(p_detection=0.9, clutter_rate=1e-5)
        phd.predict(dt=1.0)
        phd.correct(measurements)
        phd.merge()
        phd.prune()
        targets = phd.extract()
    """

    def __init__(self, **kwargs):
        config = _filters.GMPHDConfig()
        for k, v in kwargs.items():
            setattr(config, k, v)
        self._phd = _filters.GMPHD(config)
        self._transition_set = False
        self._measurement_set = False

    def set_linear_msc_dynamics(self, process_noise: float = 0.01,
                                 measurement_noise: Optional[np.ndarray] = None) -> None:
        """Configure with standard linear MSC dynamics."""
        from angle_only._angle_only_cpp import motion as _motion, measurement as _meas

        cv = _motion.ConstantVelocityMSC(process_noise)
        mm = _meas.MSCMeasurement()
        if measurement_noise is not None:
            mm.noise = np.asarray(measurement_noise, dtype=np.float64)

        self._phd.set_transition(cv.predict, cv.jacobian, cv.process_noise)
        self._phd.set_measurement(mm.predict, mm.jacobian, mm.noise)
        self._transition_set = True
        self._measurement_set = True

    def add_birth(self, components):
        """Add birth components. Each is (weight, mean, covariance)."""
        births = []
        for w, m, P in components:
            births.append(_filters.GaussianComponent(
                float(w),
                np.asarray(m, dtype=np.float64),
                np.asarray(P, dtype=np.float64)))
        self._phd.add_birth(births)

    def add_birth_from_detection(self, measurement: np.ndarray,
                                  noise: np.ndarray,
                                  range_hypotheses: Sequence[float]) -> None:
        """Add range-parameterized birth components from detection."""
        self._phd.add_birth_from_detection(
            np.asarray(measurement, dtype=np.float64),
            np.asarray(noise, dtype=np.float64),
            list(range_hypotheses))

    def predict(self, dt: float) -> None:
        self._phd.predict(dt)

    def correct(self, measurements: Sequence[np.ndarray]) -> None:
        meas = [np.asarray(m, dtype=np.float64) for m in measurements]
        self._phd.correct(meas)

    def merge(self) -> None:
        self._phd.merge()

    def prune(self) -> None:
        self._phd.prune()

    def extract(self) -> list[dict]:
        """Extract estimated targets as list of dicts."""
        targets = self._phd.extract()
        return [{"weight": t.weight,
                 "mean": np.array(t.mean),
                 "covariance": np.array(t.covariance)}
                for t in targets]

    @property
    def components(self):
        return self._phd.components

    @property
    def estimated_target_count(self) -> float:
        return self._phd.estimated_target_count

    def __repr__(self) -> str:
        n = len(self._phd.components)
        est = self._phd.estimated_target_count
        return f"GMPHD({n} components, ~{est:.1f} targets)"
