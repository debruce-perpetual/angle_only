#!/usr/bin/env python3
"""Passive ranging demo: single sensor MSCEKF tracking."""

import numpy as np
from angle_only import MSCEKF

def main():
    np.random.seed(42)

    # Target trajectory
    target_pos = np.array([500.0, 200.0, 50.0])
    target_vel = np.array([10.0, -5.0, 0.5])
    dt = 1.0
    n_steps = 50

    # Sensor at origin
    sensor_pos = np.array([0.0, 0.0, 0.0])
    noise_std = 0.001  # ~0.06 deg

    # Initialize from first detection
    rel = target_pos - sensor_pos
    az0 = np.arctan2(rel[1], rel[0])
    el0 = np.arcsin(rel[2] / np.linalg.norm(rel))
    R = np.eye(2) * noise_std**2

    ekf = MSCEKF.from_detection(azimuth=az0, elevation=el0, noise=R)

    print("Passive Ranging Demo")
    print(f"{'Step':>4} | {'Az Error (mrad)':>14} | {'El Error (mrad)':>14} | {'Est Range':>10} | {'True Range':>10}")
    print("-" * 70)

    for k in range(n_steps):
        target_pos = target_pos + target_vel * dt

        rel = target_pos - sensor_pos
        true_range = np.linalg.norm(rel)
        true_az = np.arctan2(rel[1], rel[0])
        true_el = np.arcsin(rel[2] / true_range)

        # Noisy measurement
        z = np.array([
            true_az + np.random.randn() * noise_std,
            true_el + np.random.randn() * noise_std,
        ])

        ekf.predict(dt)
        ekf.correct(z, noise=R)

        est_range = 1.0 / ekf.inv_range if ekf.inv_range > 1e-10 else float('inf')
        az_err = abs(ekf.azimuth - true_az) * 1000  # mrad
        el_err = abs(ekf.elevation - true_el) * 1000

        print(f"{k:4d} | {az_err:14.3f} | {el_err:14.3f} | {est_range:10.1f} | {true_range:10.1f}")

    print(f"\nFinal estimated range: {est_range:.1f} m (true: {true_range:.1f} m)")

if __name__ == "__main__":
    main()
