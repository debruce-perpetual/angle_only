"""Tests for Python filter wrappers."""
import pytest
import numpy as np


def test_mscekf_from_detection():
    from angle_only import MSCEKF

    ekf = MSCEKF.from_detection(azimuth=0.5, elevation=0.3)
    assert abs(ekf.azimuth - 0.5) < 1e-10
    assert abs(ekf.elevation - 0.3) < 1e-10


def test_mscekf_predict_correct():
    from angle_only import MSCEKF

    ekf = MSCEKF.from_detection(azimuth=0.0, elevation=0.0)
    ekf.predict(dt=1.0)

    z = np.array([0.01, 0.005])
    ekf.correct(z, noise=np.eye(2) * 1e-4)

    assert np.isfinite(ekf.state).all()
    assert np.isfinite(ekf.covariance).all()


def test_mscekf_covariance_shrinks():
    from angle_only import MSCEKF

    ekf = MSCEKF.from_detection(azimuth=0.0, elevation=0.0)
    ekf.predict(dt=1.0)
    trace_before = np.trace(ekf.covariance)

    ekf.correct(np.array([0.0, 0.0]), noise=np.eye(2) * 1e-4)
    trace_after = np.trace(ekf.covariance)

    assert trace_after < trace_before


def test_mscekf_distance():
    from angle_only import MSCEKF

    ekf = MSCEKF.from_detection(azimuth=0.0, elevation=0.0)
    d_close = ekf.distance(np.array([0.001, 0.001]))
    d_far = ekf.distance(np.array([1.0, 1.0]))
    assert d_close < d_far


def test_mscekf_likelihood():
    from angle_only import MSCEKF

    ekf = MSCEKF.from_detection(azimuth=0.0, elevation=0.0)
    l_close = ekf.likelihood(np.array([0.001, 0.001]))
    l_far = ekf.likelihood(np.array([1.0, 1.0]))
    assert l_close > l_far


def test_mscekf_repr():
    from angle_only import MSCEKF

    ekf = MSCEKF.from_detection(azimuth=0.5, elevation=0.3)
    r = repr(ekf)
    assert "MSCEKF" in r
    assert "0.5" in r


def test_mscekf_smoother():
    from angle_only import MSCEKF

    ekf = MSCEKF.from_detection(azimuth=0.0, elevation=0.0)
    ekf.set_store_history(True)

    for i in range(5):
        ekf.predict(1.0)
        z = np.array([0.01 * (i + 1), 0.005 * (i + 1)])
        ekf.correct(z, noise=np.eye(2) * 1e-4)

    smoothed = ekf.smooth()
    assert len(smoothed) == 5


def test_gmphd_basic():
    from angle_only import GMPHD

    phd = GMPHD(p_detection=0.9, clutter_rate=1e-5)
    assert phd.estimated_target_count == 0.0


def test_gmphd_repr():
    from angle_only import GMPHD

    phd = GMPHD()
    r = repr(phd)
    assert "GMPHD" in r
