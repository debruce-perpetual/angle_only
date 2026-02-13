"""Tests for coordinate transforms."""
import pytest
import numpy as np


def test_spherical_to_cartesian():
    from angle_only import coords

    p = coords.spherical_to_cartesian(0.0, 0.0, 1.0)
    np.testing.assert_allclose(p, [1.0, 0.0, 0.0], atol=1e-15)


def test_wrap_to_pi():
    from angle_only import coords

    assert abs(coords.wrap_to_pi(0.0)) < 1e-15
    assert abs(coords.wrap_to_pi(7.0) - (7.0 - 2 * np.pi)) < 1e-10


def test_az_el_roundtrip():
    from angle_only import coords

    az, el = 0.8, 0.4
    v = coords.az_el_to_unit_vector(az, el)
    az2, el2 = coords.unit_vector_to_az_el(v)
    assert abs(az - az2) < 1e-12
    assert abs(el - el2) < 1e-12
