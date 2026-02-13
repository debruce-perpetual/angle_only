function d = mahalanobis_distance(innovation, S)
%MAHALANOBIS_DISTANCE Compute Mahalanobis distance.
%   d = aot.mahalanobis_distance(innovation, S)
%
%   Inputs:
%     innovation — Nx1 innovation vector (z - z_predicted)
%     S          — NxN innovation covariance
%
%   Output:
%     d — Mahalanobis distance (scalar)

d = angle_only_mex('mahalanobis_distance', innovation(:), S);
end
