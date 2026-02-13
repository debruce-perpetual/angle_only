"""Integration test: passive ranging scenario.

Two sensors observe a target moving in a straight line.
The MSCEKF on each sensor converges to the true trajectory.
"""
import pytest
import numpy as np


def test_passive_ranging_convergence():
    from angle_only import MSCEKF, core, coords

    np.random.seed(42)

    # Target trajectory: constant velocity
    target_start = np.array([500.0, 200.0, 50.0])
    target_vel = np.array([10.0, -5.0, 0.5])
    dt = 1.0
    n_steps = 20

    # Sensor at origin
    sensor_pos = np.array([0.0, 0.0, 0.0])
    noise_std = 0.001  # ~0.06 deg

    # Initialize EKF from first detection
    pos0 = target_start
    rel = pos0 - sensor_pos
    az0 = float(np.arctan2(rel[1], rel[0]))
    el0 = float(np.arcsin(rel[2] / np.linalg.norm(rel)))
    R = np.eye(2) * noise_std**2

    ekf = MSCEKF.from_detection(azimuth=az0, elevation=el0, noise=R)

    errors = []
    for k in range(n_steps):
        target_pos = target_start + target_vel * (k + 1) * dt
        rel = target_pos - sensor_pos
        r = np.linalg.norm(rel)
        true_az = float(np.arctan2(rel[1], rel[0]))
        true_el = float(np.arcsin(rel[2] / r))

        # Noisy measurement
        z_az = true_az + np.random.randn() * noise_std
        z_el = true_el + np.random.randn() * noise_std

        ekf.predict(dt)
        ekf.correct(np.array([z_az, z_el]), noise=R)

        # Track angular error
        az_err = abs(ekf.azimuth - true_az)
        el_err = abs(ekf.elevation - true_el)
        errors.append(max(az_err, el_err))

    # Angular errors should be small (< 10 mrad) after convergence
    assert errors[-1] < 0.01, f"Final angular error too large: {errors[-1]}"
    # Error should generally decrease
    assert np.mean(errors[-5:]) < np.mean(errors[:5])


def test_two_sensor_triangulation():
    """Two sensors triangulate a target at each step."""
    from angle_only import MSCEKF, triangulate_los, core
    import numpy as np

    target = np.array([200.0, 100.0, 30.0])

    sensors = [
        core.SensorPose(),
        core.SensorPose(),
    ]
    sensors[0].position = np.array([0.0, 0.0, 0.0])
    sensors[0].orientation = np.eye(3)
    sensors[1].position = np.array([0.0, 300.0, 0.0])
    sensors[1].orientation = np.eye(3)

    detections = []
    for s in sensors:
        rel = target - s.position
        d = core.Detection()
        d.azimuth = float(np.arctan2(rel[1], rel[0]))
        d.elevation = float(np.arcsin(rel[2] / np.linalg.norm(rel)))
        d.noise = np.eye(2) * 1e-4
        detections.append(d)

    result = triangulate_los(sensors, detections)
    assert result["valid"]
    error = np.linalg.norm(result["position"] - target)
    assert error < 1.0, f"Triangulation error too large: {error}"
