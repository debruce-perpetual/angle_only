"""Tests for Python fusion wrappers."""
import pytest
import numpy as np


def test_triangulate_los():
    from angle_only import triangulate_los, core

    target = np.array([100.0, 50.0, 0.0])

    m1 = core.LOSMeasurement()
    m1.origin = np.array([0.0, 0.0, 0.0])
    m1.direction = target / np.linalg.norm(target)
    m1.noise = np.eye(2) * 1e-4

    origin2 = np.array([0.0, 100.0, 0.0])
    dir2 = (target - origin2)
    m2 = core.LOSMeasurement()
    m2.origin = origin2
    m2.direction = dir2 / np.linalg.norm(dir2)
    m2.noise = np.eye(2) * 1e-4

    result = triangulate_los([m1, m2])
    assert result["valid"]
    assert np.linalg.norm(result["position"] - target) < 1.0


def test_triangulate_from_sensors():
    from angle_only import triangulate_los, core

    target = np.array([100.0, 0.0, 0.0])

    s1 = core.SensorPose()
    s1.position = np.array([0.0, -50.0, 0.0])
    s1.orientation = np.eye(3)

    s2 = core.SensorPose()
    s2.position = np.array([0.0, 50.0, 0.0])
    s2.orientation = np.eye(3)

    dir1 = (target - s1.position)
    dir2 = (target - s2.position)

    d1 = core.Detection()
    d1.azimuth = float(np.arctan2(dir1[1], dir1[0]))
    d1.elevation = float(np.arcsin(dir1[2] / np.linalg.norm(dir1)))
    d1.noise = np.eye(2) * 1e-4

    d2 = core.Detection()
    d2.azimuth = float(np.arctan2(dir2[1], dir2[0]))
    d2.elevation = float(np.arcsin(dir2[2] / np.linalg.norm(dir2)))
    d2.noise = np.eye(2) * 1e-4

    result = triangulate_los([s1, s2], [d1, d2])
    assert result["valid"]
    assert np.linalg.norm(result["position"] - target) < 2.0
