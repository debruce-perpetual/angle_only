# Python API Reference

## Installation

```bash
pip install angle-only
```

## Top-Level API

```python
import angle_only

# Filters
ekf = angle_only.MSCEKF.from_detection(azimuth=0.5, elevation=0.3)
phd = angle_only.GMPHD(p_detection=0.9)

# Fusion
result = angle_only.triangulate_los(sensors, detections)
fuser = angle_only.StaticDetectionFuser(max_distance=0.1)

# Submodules
angle_only.core       # types, constants
angle_only.coords     # coordinate transforms
angle_only.motion     # motion models
angle_only.measurement # measurement models
angle_only.filters    # C++ filter bindings
angle_only.fusion     # fusion algorithms
angle_only.association # data association
angle_only.gpu        # GPU dispatch
```

## MSCEKF

```python
from angle_only import MSCEKF

ekf = MSCEKF.from_detection(azimuth=0.5, elevation=0.3,
                              noise=np.eye(2) * 1e-4)
ekf.predict(dt=1.0)
ekf.correct(np.array([0.51, 0.29]), noise=np.eye(2) * 1e-4)

ekf.state          # np.ndarray (6,)
ekf.covariance     # np.ndarray (6, 6)
ekf.azimuth        # float
ekf.elevation      # float
ekf.inv_range      # float
ekf.distance(z)    # Mahalanobis distance
ekf.likelihood(z)  # Gaussian likelihood
ekf.smooth()       # RTS smoother
```

## GMPHD

```python
from angle_only import GMPHD

phd = GMPHD(p_detection=0.9, clutter_rate=1e-5)
phd.add_birth_from_detection(z, R, range_hypotheses=[100, 500, 1000])
phd.predict(dt=1.0)
phd.correct(measurements)
phd.merge()
phd.prune()
targets = phd.extract()  # list of {weight, mean, covariance}
```

## GPU

```python
from angle_only import gpu

gpu.cuda_available()    # bool
gpu.device_count()      # int
gpu.should_use_gpu(64)  # bool
```
