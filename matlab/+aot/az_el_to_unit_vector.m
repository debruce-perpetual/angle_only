function dir = az_el_to_unit_vector(az, el)
%AZ_EL_TO_UNIT_VECTOR Convert azimuth and elevation to unit direction vector.
%   dir = aot.az_el_to_unit_vector(az, el)
%
%   Inputs:
%     az — azimuth angle (radians)
%     el — elevation angle (radians)
%
%   Output:
%     dir — 3x1 unit direction vector

dir = angle_only_mex('az_el_to_unit_vector', az, el);
end
