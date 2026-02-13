function setup()
%SETUP Add angle_only MATLAB bindings to the path.
%   Run this once per MATLAB session:
%     run('path/to/angle_only/matlab/setup.m')

this_dir = fileparts(mfilename('fullpath'));

% Add the matlab directory (contains +aot package and MEX binary)
addpath(this_dir);

% Verify MEX binary exists
if exist('angle_only_mex', 'file') ~= 3
    warning('aot:setup', ...
        ['angle_only_mex MEX binary not found on path.\n' ...
         'Build it with: cmake -B build -DAOT_BUILD_MATLAB=ON && cmake --build build\n' ...
         'Then ensure the build output directory is on the MATLAB path.']);
else
    fprintf('angle_only MATLAB bindings loaded successfully.\n');
    fprintf('  Example: ekf = aot.initcvmscekf(det)\n');
end
end
