function beta = jpda_probabilities(likelihood_matrix, p_detection, p_gate, clutter_density)
%JPDA_PROBABILITIES Compute JPDA association probabilities.
%   beta = aot.jpda_probabilities(likelihood_matrix)
%   beta = aot.jpda_probabilities(likelihood_matrix, p_detection, p_gate, clutter_density)
%
%   Inputs:
%     likelihood_matrix — MxN matrix: likelihood(i,j) = P(z_j | track_i)
%     p_detection       — (optional) probability of detection (default: 0.9)
%     p_gate            — (optional) probability of gated measurement (default: 0.99)
%     clutter_density   — (optional) clutter spatial density (default: 1e-6)
%
%   Output:
%     beta — Mx(N+1) matrix: beta(i,j) = P(measurement j from track i)
%            beta(i, N+1) = P(track i has no measurement)

args = {likelihood_matrix};
if nargin > 1, args{end+1} = p_detection; end
if nargin > 2, args{end+1} = p_gate; end
if nargin > 3, args{end+1} = clutter_density; end

beta = angle_only_mex('jpda_probabilities', args{:});
end
