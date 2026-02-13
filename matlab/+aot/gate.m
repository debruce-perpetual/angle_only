function indices = gate(predicted_measurement, measurements, S, threshold)
%GATE Ellipsoidal gating using Mahalanobis distance.
%   indices = aot.gate(predicted_measurement, measurements, S, threshold)
%
%   Inputs:
%     predicted_measurement — Mx1 predicted measurement vector
%     measurements          — NxM matrix (each row is a measurement)
%     S                     — MxM innovation covariance
%     threshold             — gating threshold (chi-squared)
%
%   Output:
%     indices — vector of 1-indexed measurement indices that pass the gate

indices = angle_only_mex('gate', predicted_measurement(:), measurements, S, threshold);
end
