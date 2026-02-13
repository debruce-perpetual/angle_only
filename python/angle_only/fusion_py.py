"""Pythonic wrappers for fusion algorithms."""

from __future__ import annotations

import numpy as np
from typing import Optional, Sequence

try:
    from angle_only._angle_only_cpp import core, fusion as _fusion
except ImportError:
    _fusion = None
    core = None


def triangulate_los(sensors, detections=None, *, measurements=None):
    """Triangulate target position from line-of-sight measurements.

    Can be called as:
        triangulate_los(los_measurements)
        triangulate_los(sensors, detections)

    Returns:
        dict with 'position', 'covariance', 'residual', 'valid' keys
    """
    if measurements is not None:
        result = _fusion.triangulate_los(measurements)
    elif detections is not None:
        result = _fusion.triangulate_los(sensors, detections)
    else:
        result = _fusion.triangulate_los(sensors)

    return {
        "position": np.array(result.position),
        "covariance": np.array(result.covariance),
        "residual": result.residual,
        "valid": result.valid,
    }


class StaticDetectionFuser:
    """Fuse detections from multiple sensors into position estimates.

    Example::

        fuser = StaticDetectionFuser(max_distance=0.1, min_detections=2)
        fused = fuser.fuse(sensors, detections_per_sensor)
    """

    def __init__(self, max_distance: float = 0.1,
                 min_detections: int = 2,
                 use_triangulation: bool = True):
        config = _fusion.StaticDetectionFuserConfig()
        config.max_distance = max_distance
        config.min_detections = min_detections
        config.use_triangulation = use_triangulation
        self._fuser = _fusion.StaticDetectionFuser(config)

    def fuse(self, sensors, detections_per_sensor) -> list[dict]:
        """Fuse detections and return list of fused position estimates."""
        results = self._fuser.fuse(sensors, detections_per_sensor)
        return [{
            "position": np.array(r.position),
            "covariance": np.array(r.covariance),
            "time": r.time,
            "sensor_ids": list(r.sensor_ids),
        } for r in results]
