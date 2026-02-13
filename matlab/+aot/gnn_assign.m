function [assignments, unassigned_tracks, unassigned_meas, cost] = gnn_assign(cost_matrix, gate_threshold)
%GNN_ASSIGN Global Nearest Neighbor assignment using Hungarian algorithm.
%   [assignments, unassigned_tracks, unassigned_meas, cost] = aot.gnn_assign(cost_matrix)
%   [...] = aot.gnn_assign(cost_matrix, gate_threshold)
%
%   Inputs:
%     cost_matrix    — MxN cost matrix (tracks x measurements)
%     gate_threshold — (optional) max cost for valid assignment (default: 1e10)
%
%   Outputs:
%     assignments       — Kx2 matrix of [track_idx, meas_idx] (1-indexed)
%     unassigned_tracks — vector of unassigned track indices (1-indexed)
%     unassigned_meas   — vector of unassigned measurement indices (1-indexed)
%     cost              — total assignment cost

if nargin < 2
    [assignments, unassigned_tracks, unassigned_meas, cost] = ...
        angle_only_mex('gnn_assign', cost_matrix);
else
    [assignments, unassigned_tracks, unassigned_meas, cost] = ...
        angle_only_mex('gnn_assign', cost_matrix, gate_threshold);
end
end
