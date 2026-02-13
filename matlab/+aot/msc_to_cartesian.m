function pos = msc_to_cartesian(az, el, inv_range)
%MSC_TO_CARTESIAN Convert Modified Spherical Coordinates to Cartesian.
%   pos = aot.msc_to_cartesian(az, el, inv_range)
%
%   Inputs:
%     az        — azimuth angle (radians)
%     el        — elevation angle (radians)
%     inv_range — inverse range (1/r)
%
%   Output:
%     pos — 3x1 [x; y; z]

pos = angle_only_mex('msc_to_cartesian', az, el, inv_range);
end
