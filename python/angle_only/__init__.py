"""Angle-only tracking library — Python interface.

Provides MATLAB-compatible angle-only tracking algorithms including:
- MSCEKF (Modified Spherical Coordinates Extended Kalman Filter)
- GM-PHD (Gaussian Mixture Probability Hypothesis Density) filter
- LOS triangulation and static detection fusion
- GNN, JPDA data association
- Optional GPU acceleration via CUDA
"""

try:
    from angle_only._angle_only_cpp import core, coords, motion, measurement, filters, fusion, association, gpu
except ImportError:
    # Allow import even if C++ extension is not built (for docs, type checking)
    import warnings
    warnings.warn("C++ extension not available. Install with: pip install .", stacklevel=2)
    core = coords = motion = measurement = filters = fusion = association = gpu = None  # type: ignore

from angle_only.filters_py import MSCEKF, GMPHD
from angle_only.fusion_py import triangulate_los, StaticDetectionFuser
from angle_only.compat import *  # noqa: F401,F403 — MATLAB-compatible aliases

__version__ = "0.1.0"

__all__ = [
    "MSCEKF", "GMPHD",
    "triangulate_los", "StaticDetectionFuser",
    "core", "coords", "motion", "measurement", "filters", "fusion", "association", "gpu",
]
