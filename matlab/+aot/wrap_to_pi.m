function wrapped = wrap_to_pi(angle)
%WRAP_TO_PI Wrap angle to [-pi, pi].
%   wrapped = aot.wrap_to_pi(angle)
%
%   Input:
%     angle — angle in radians
%
%   Output:
%     wrapped — angle wrapped to [-pi, pi]

wrapped = angle_only_mex('wrap_to_pi', angle);
end
