# C++ API Reference

## Quick Include

```cpp
#include <angle_only/angle_only.hpp>  // everything
// or individual headers:
#include <angle_only/filters/msc_ekf.hpp>
```

## Namespace: `aot`

### Core Types (`angle_only/core/types.hpp`)

| Type | Description |
|------|-------------|
| `Vec2, Vec3, Vec6` | Fixed-size column vectors |
| `Mat2, Mat3, Mat6` | Fixed-size row-major matrices |
| `VecX, MatXR` | Dynamic-size types |

### MSCState (`angle_only/core/state.hpp`)

State vector: `[az, el, 1/r, az_rate, el_rate, inv_range_rate]`

### Filters (`angle_only/filters/`)

#### MSCEKF

```cpp
auto ekf = aot::filters::initcvmscekf(detection);
ekf.predict(dt);
ekf.correct(measurement, R);
double d = ekf.distance(z, R);
double l = ekf.likelihood(z, R);
auto smoothed = ekf.smooth();  // RTS smoother
```

#### GMPHD

```cpp
aot::filters::GMPHD::Config config;
config.p_detection = 0.9;
aot::filters::GMPHD phd(config);
phd.add_birth(components);
phd.predict(dt);
phd.correct(measurements);
phd.merge();
phd.prune();
auto targets = phd.extract();
```

### Motion Models (`angle_only/motion/`)

| Class | State Dim | Description |
|-------|-----------|-------------|
| `ConstantVelocityMSC` | 6 | MSC coordinates |
| `ConstantVelocity` | 6 | Cartesian |
| `ConstantAcceleration` | 9 | Cartesian |
| `CoordinatedTurn` | 7 | Cartesian + turn rate |

### Fusion (`angle_only/fusion/`)

```cpp
auto result = aot::fusion::triangulate_los(los_measurements);
// result.position, result.covariance, result.valid
```

### Association (`angle_only/association/`)

```cpp
auto result = aot::association::gnn_assign(cost_matrix, gate_threshold);
auto jpda = aot::association::jpda_probabilities(likelihood_matrix);
```
