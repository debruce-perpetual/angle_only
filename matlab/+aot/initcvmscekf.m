function ekf = initcvmscekf(det, inv_range, inv_range_std)
%INITCVMSCEKF Initialize a constant-velocity MSC-EKF from a detection.
%   ekf = aot.initcvmscekf(det)
%   ekf = aot.initcvmscekf(det, inv_range)
%   ekf = aot.initcvmscekf(det, inv_range, inv_range_std)
%
%   det is a struct with fields:
%     azimuth    — bearing angle (radians)
%     elevation  — elevation angle (radians)
%     noise      — 2x2 measurement noise covariance
%     time       — (optional) timestamp
%     sensor_id  — (optional) sensor identifier
%
%   inv_range     — initial inverse range estimate (default: 0.01)
%   inv_range_std — initial inverse range std deviation (default: 0.05)
%
%   Returns an aot.MSCEKF object.

if nargin < 2, inv_range = 0.01; end
if nargin < 3, inv_range_std = 0.05; end

ekf = aot.MSCEKF(det, inv_range, inv_range_std);
end
