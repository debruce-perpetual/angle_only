function [az, el, inv_range] = cartesian_to_msc(pos)
%CARTESIAN_TO_MSC Convert Cartesian position to Modified Spherical Coordinates.
%   [az, el, inv_range] = aot.cartesian_to_msc(pos)
%
%   Input:
%     pos — 3x1 [x; y; z]
%
%   Outputs:
%     az        — azimuth angle (radians)
%     el        — elevation angle (radians)
%     inv_range — inverse range (1/r)

[az, el, inv_range] = angle_only_mex('cartesian_to_msc', pos(:));
end
