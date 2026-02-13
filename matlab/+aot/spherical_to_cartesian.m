function pos = spherical_to_cartesian(az, el, r)
%SPHERICAL_TO_CARTESIAN Convert spherical coordinates to Cartesian.
%   pos = aot.spherical_to_cartesian(az, el, r)
%
%   Inputs:
%     az — azimuth angle (radians)
%     el — elevation angle (radians)
%     r  — range
%
%   Output:
%     pos — 3x1 [x; y; z]

pos = angle_only_mex('spherical_to_cartesian', az, el, r);
end
