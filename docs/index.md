# Angle-Only Tracking Library

GPU-accelerated angle-only tracking library with C++20 core and Python bindings.

## Features

- **MSCEKF** — Modified Spherical Coordinates Extended Kalman Filter for angle-only tracking
- **GM-PHD** — Gaussian Mixture PHD filter for multi-target tracking
- **LOS Triangulation** — Linear least-squares line-of-sight triangulation
- **Data Association** — GNN (Hungarian), JPDA, Mahalanobis gating
- **Motion Models** — Constant Velocity MSC, CV, CA, Coordinated Turn
- **GPU Acceleration** — Optional CUDA batch operations with automatic CPU fallback
- **Python Bindings** — Pythonic wrappers with numpy interop via pybind11

## Quick Start

### C++

```cpp
#include <angle_only/angle_only.hpp>

// Initialize filter from angular detection
aot::Detection det{.azimuth = 0.5, .elevation = 0.3};
auto ekf = aot::filters::initcvmscekf(det);

// Predict-correct loop
ekf.predict(1.0);
aot::Vec2 z; z << 0.51, 0.29;
ekf.correct(z, aot::Mat2::Identity() * 1e-4);
```

### Python

```python
from angle_only import MSCEKF
import numpy as np

ekf = MSCEKF.from_detection(azimuth=0.5, elevation=0.3)
ekf.predict(dt=1.0)
ekf.correct(np.array([0.51, 0.29]), noise=np.eye(2) * 1e-4)
```
