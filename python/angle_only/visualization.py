"""3D visualization utilities using Plotly.

Generates interactive HTML plots for tracking scenarios.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Sequence


def _check_plotly():
    try:
        import plotly.graph_objects as go
        return go
    except ImportError:
        raise ImportError(
            "plotly is required for visualization. Install with: "
            "pip install plotly"
        )


def plot_tracking_scenario(
    true_positions: np.ndarray,
    estimated_positions: Optional[np.ndarray] = None,
    sensor_positions: Optional[np.ndarray] = None,
    detections: Optional[list] = None,
    covariances: Optional[np.ndarray] = None,
    title: str = "Angle-Only Tracking Scenario",
    output_file: Optional[str] = None,
) -> "plotly.graph_objects.Figure":
    """Create an interactive 3D tracking scenario plot.

    Args:
        true_positions: (N, 3) array of true target positions
        estimated_positions: (N, 3) array of estimated positions
        sensor_positions: (M, 3) array of sensor locations
        detections: list of detection dicts for LOS line visualization
        covariances: (N, 3, 3) array of position covariances
        title: plot title
        output_file: if provided, save HTML to this path

    Returns:
        Plotly Figure object
    """
    go = _check_plotly()
    fig = go.Figure()

    # True trajectory
    if true_positions is not None:
        tp = np.asarray(true_positions)
        fig.add_trace(go.Scatter3d(
            x=tp[:, 0], y=tp[:, 1], z=tp[:, 2],
            mode='lines+markers',
            name='True Position',
            line=dict(color='blue', width=3),
            marker=dict(size=2),
        ))

    # Estimated trajectory
    if estimated_positions is not None:
        ep = np.asarray(estimated_positions)
        fig.add_trace(go.Scatter3d(
            x=ep[:, 0], y=ep[:, 1], z=ep[:, 2],
            mode='lines+markers',
            name='Estimated Position',
            line=dict(color='red', width=3, dash='dash'),
            marker=dict(size=2),
        ))

    # Sensor positions
    if sensor_positions is not None:
        sp = np.asarray(sensor_positions)
        fig.add_trace(go.Scatter3d(
            x=sp[:, 0], y=sp[:, 1], z=sp[:, 2],
            mode='markers+text',
            name='Sensors',
            marker=dict(size=8, color='green', symbol='diamond'),
            text=[f'S{i}' for i in range(len(sp))],
        ))

    # LOS lines from sensors
    if detections is not None and sensor_positions is not None:
        sp = np.asarray(sensor_positions)
        for i, det_list in enumerate(detections):
            for det in (det_list if isinstance(det_list, list) else [det_list]):
                if hasattr(det, 'azimuth'):
                    az, el = det.azimuth, det.elevation
                else:
                    az, el = det.get('azimuth', 0), det.get('elevation', 0)

                direction = np.array([
                    np.cos(el) * np.cos(az),
                    np.cos(el) * np.sin(az),
                    np.sin(el),
                ])
                end = sp[i] + direction * 200  # extend LOS line
                fig.add_trace(go.Scatter3d(
                    x=[sp[i, 0], end[0]],
                    y=[sp[i, 1], end[1]],
                    z=[sp[i, 2], end[2]],
                    mode='lines',
                    line=dict(color='gray', width=1, dash='dot'),
                    showlegend=False,
                ))

    # Covariance ellipsoids (sample a few points)
    if covariances is not None and estimated_positions is not None:
        ep = np.asarray(estimated_positions)
        step = max(1, len(ep) // 10)  # show ~10 ellipsoids
        for k in range(0, len(ep), step):
            P = covariances[k]
            eigvals, eigvecs = np.linalg.eigh(P)
            radii = 2.0 * np.sqrt(np.maximum(eigvals, 0))

            # Generate sphere points
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 10)
            x = radii[0] * np.outer(np.cos(u), np.sin(v))
            y = radii[1] * np.outer(np.sin(u), np.sin(v))
            z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

            # Rotate and translate
            for i in range(len(u)):
                for j in range(len(v)):
                    pt = eigvecs @ np.array([x[i, j], y[i, j], z[i, j]])
                    x[i, j] = pt[0] + ep[k, 0]
                    y[i, j] = pt[1] + ep[k, 1]
                    z[i, j] = pt[2] + ep[k, 2]

            fig.add_trace(go.Surface(
                x=x, y=y, z=z,
                opacity=0.2, colorscale=[[0, 'red'], [1, 'red']],
                showscale=False, showlegend=False,
            ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data',
        ),
        width=1000, height=700,
    )

    if output_file:
        fig.write_html(output_file)

    return fig
