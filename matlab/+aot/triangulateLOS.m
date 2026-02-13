function result = triangulateLOS(origins, directions, noise)
%TRIANGULATELOS Triangulate target position from multiple lines of sight.
%   result = aot.triangulateLOS(origins, directions)
%   result = aot.triangulateLOS(origins, directions, noise)
%
%   Inputs:
%     origins    — Nx3 matrix of sensor positions [x, y, z]
%     directions — Nx3 matrix of unit direction vectors
%     noise      — (optional) 2x2 angular noise covariance,
%                  or cell array of N 2x2 matrices
%
%   Output struct:
%     position   — 3x1 estimated target position
%     covariance — 3x3 position uncertainty
%     residual   — sum of squared residuals
%     valid      — true if triangulation succeeded

if nargin < 3
    [pos, cov, res, valid] = angle_only_mex('triangulate_los', origins, directions);
else
    [pos, cov, res, valid] = angle_only_mex('triangulate_los', origins, directions, noise);
end

result.position = pos;
result.covariance = cov;
result.residual = res;
result.valid = valid;
end
