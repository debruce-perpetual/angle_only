function [az, el, r] = cartesian_to_spherical(pos)
%CARTESIAN_TO_SPHERICAL Convert Cartesian position to spherical coordinates.
%   [az, el, r] = aot.cartesian_to_spherical(pos)
%
%   Input:
%     pos — 3x1 [x; y; z]
%
%   Outputs:
%     az — azimuth angle (radians)
%     el — elevation angle (radians)
%     r  — range

[az, el, r] = angle_only_mex('cartesian_to_spherical', pos(:));
end
