function [assignments, unassigned_tracks, unassigned_meas, cost] = auction_assign(cost_matrix, epsilon, gate_threshold)
%AUCTION_ASSIGN Auction algorithm for assignment (faster for sparse problems).
%   [assignments, unassigned_tracks, unassigned_meas, cost] = aot.auction_assign(cost_matrix)
%   [...] = aot.auction_assign(cost_matrix, epsilon)
%   [...] = aot.auction_assign(cost_matrix, epsilon, gate_threshold)
%
%   Inputs:
%     cost_matrix    — MxN cost matrix (tracks x measurements)
%     epsilon        — (optional) auction precision (default: 1e-6)
%     gate_threshold — (optional) max cost for valid assignment (default: 1e10)
%
%   Outputs:
%     assignments       — Kx2 matrix of [track_idx, meas_idx] (1-indexed)
%     unassigned_tracks — vector of unassigned track indices (1-indexed)
%     unassigned_meas   — vector of unassigned measurement indices (1-indexed)
%     cost              — total assignment cost

args = {cost_matrix};
if nargin > 1, args{end+1} = epsilon; end
if nargin > 2, args{end+1} = gate_threshold; end

[assignments, unassigned_tracks, unassigned_meas, cost] = ...
    angle_only_mex('auction_assign', args{:});
end
