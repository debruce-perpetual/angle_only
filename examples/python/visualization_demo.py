#!/usr/bin/env python3
"""Visualization demo: 3D tracking scenario with Plotly."""

import numpy as np

def main():
    try:
        from angle_only.visualization import plot_tracking_scenario
    except ImportError:
        print("Plotly required. Install with: pip install plotly")
        return

    np.random.seed(42)

    # Generate target trajectory (circular motion)
    t = np.linspace(0, 20, 100)
    radius = 500.0
    true_positions = np.column_stack([
        radius * np.cos(0.1 * t),
        radius * np.sin(0.1 * t),
        50.0 + 5.0 * t
    ])

    # Simulated estimated positions (with noise)
    estimated_positions = true_positions + np.random.randn(*true_positions.shape) * 10

    # Sensor positions
    sensor_positions = np.array([
        [0, 0, 0],
        [200, 0, 0],
        [0, 200, 0],
    ], dtype=float)

    # Covariances (shrinking over time)
    covariances = np.zeros((len(t), 3, 3))
    for i in range(len(t)):
        scale = max(10.0, 100.0 - i)
        covariances[i] = np.diag([scale, scale, scale * 0.5])

    fig = plot_tracking_scenario(
        true_positions=true_positions,
        estimated_positions=estimated_positions,
        sensor_positions=sensor_positions,
        covariances=covariances,
        title="Angle-Only Tracking Demo",
        output_file="tracking_demo.html",
    )

    print("Saved tracking_demo.html")

if __name__ == "__main__":
    main()
