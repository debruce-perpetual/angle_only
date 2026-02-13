function threshold = chi2_gate(dimension, confidence)
%CHI2_GATE Chi-squared gate threshold for given dimension and confidence.
%   threshold = aot.chi2_gate(dimension)
%   threshold = aot.chi2_gate(dimension, confidence)
%
%   Inputs:
%     dimension  — measurement dimension (e.g. 2 for az/el)
%     confidence — (optional) confidence level (default: 0.99)
%
%   Output:
%     threshold — chi-squared threshold value

if nargin < 2
    threshold = angle_only_mex('chi2_gate', dimension);
else
    threshold = angle_only_mex('chi2_gate', dimension, confidence);
end
end
