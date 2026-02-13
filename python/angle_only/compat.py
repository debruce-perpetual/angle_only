"""MATLAB-compatible aliases for migration from Sensor Fusion and Tracking Toolbox.

These aliases map MATLAB function names to their Python equivalents:
    trackingMSCEKF -> MSCEKF
    constvelmsc -> ConstantVelocityMSC
    cvmeasmsc -> MSCMeasurement
    initcvmscekf -> MSCEKF.from_detection
    triangulateLOS -> triangulate_los
    staticDetectionFuser -> StaticDetectionFuser
    gmphd -> GMPHD
"""

try:
    from angle_only._angle_only_cpp import motion, measurement, filters, fusion

    # Motion models
    constvelmsc = motion.ConstantVelocityMSC
    constvel = motion.ConstantVelocity
    constacc = motion.ConstantAcceleration
    constturn = motion.CoordinatedTurn

    # Measurement models
    cvmeasmsc = measurement.MSCMeasurement
    sphericalmeas = measurement.SphericalMeasurement

    # Filter initialization
    initcvmscekf = filters.initcvmscekf

    # Fusion
    triangulateLOS = fusion.triangulate_los

except ImportError:
    pass

# Re-export from pythonic wrappers
from angle_only.filters_py import MSCEKF as trackingMSCEKF  # noqa: F401
from angle_only.filters_py import GMPHD as gmphd  # noqa: F401
from angle_only.fusion_py import StaticDetectionFuser as staticDetectionFuser  # noqa: F401
