# MATLAB Migration Guide

## Function Mapping

| MATLAB | C++ | Python |
|--------|-----|--------|
| `trackingMSCEKF` | `aot::filters::MSCEKF` | `angle_only.MSCEKF` |
| `predict(ekf, dt)` | `ekf.predict(dt)` | `ekf.predict(dt)` |
| `correct(ekf, z)` | `ekf.correct(z, R)` | `ekf.correct(z, noise=R)` |
| `distance(ekf, z)` | `ekf.distance(z, R)` | `ekf.distance(z)` |
| `likelihood(ekf, z)` | `ekf.likelihood(z, R)` | `ekf.likelihood(z)` |
| `constvelmsc` | `aot::motion::ConstantVelocityMSC` | `angle_only.motion.ConstantVelocityMSC` |
| `cvmeasmsc` | `aot::measurement::MSCMeasurement` | `angle_only.measurement.MSCMeasurement` |
| `initcvmscekf` | `aot::filters::initcvmscekf()` | `MSCEKF.from_detection()` |
| `triangulateLOS` | `aot::fusion::triangulate_los()` | `angle_only.triangulate_los()` |
| `staticDetectionFuser` | `aot::fusion::StaticDetectionFuser` | `angle_only.StaticDetectionFuser` |
| `gmphd` | `aot::filters::GMPHD` | `angle_only.GMPHD` |

## MATLAB Compatibility Module

```python
from angle_only.compat import *

# Use MATLAB-style names
ekf = trackingMSCEKF.from_detection(azimuth=0.5, elevation=0.3)
model = constvelmsc(1.0)
result = triangulateLOS(measurements)
```

## Key Differences

1. **Explicit noise**: C++/Python require explicit noise covariance `R` parameter
2. **No object handles**: C++ uses value semantics, Python uses wrapper objects
3. **Angle units**: Always radians (MATLAB sometimes uses degrees)
4. **State access**: `ekf.state` (property) instead of `ekf.State`
