# Examples

## Passive Ranging (Single Sensor)

Track a target using angle-only measurements from a single sensor. The MSCEKF estimates both angular state and range through observability from target motion.

```python
from angle_only import MSCEKF
import numpy as np

ekf = MSCEKF.from_detection(azimuth=0.5, elevation=0.3)

for measurement in measurements:
    ekf.predict(dt=1.0)
    ekf.correct(measurement, noise=np.eye(2) * 1e-4)
    print(f"Range estimate: {1.0 / ekf.inv_range:.0f} m")
```

Run the full demo:
```bash
python examples/python/passive_ranging_demo.py
```

## Multi-Sensor Triangulation

Two or more sensors triangulate a target position using LOS measurements.

```python
from angle_only import triangulate_los, core

# Set up sensors and detections
result = triangulate_los(sensors, detections)
print(f"Position: {result['position']}")
```

## GM-PHD Multi-Target Tracking

Track multiple targets with unknown number using the GM-PHD filter.

```python
from angle_only import GMPHD

phd = GMPHD(p_detection=0.9, clutter_rate=1e-5)

for measurements in measurement_sequence:
    phd.predict(dt=1.0)
    phd.correct(measurements)
    phd.merge()
    phd.prune()
    targets = phd.extract()
    print(f"{len(targets)} targets detected")
```

## 3D Visualization

Generate interactive HTML plots of tracking scenarios.

```python
from angle_only.visualization import plot_tracking_scenario

fig = plot_tracking_scenario(
    true_positions=truth,
    estimated_positions=estimates,
    sensor_positions=sensors,
    output_file="tracking.html",
)
```

Run: `python examples/python/visualization_demo.py`
