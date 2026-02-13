%% Passive Ranging Demo — angle_only MATLAB MEX bindings
%
% Demonstrates single-sensor passive ranging using the MSC-EKF.
% A sensor on a manoeuvring platform tracks a stationary target using
% angle-only measurements. Range becomes observable through geometry change.
%
% Run setup first:
%   run('path/to/angle_only/matlab/setup.m')

%% Parameters
rng(42);

% Target position (stationary)
target_pos = [5000; 3000; 1000];  % metres

% Sensor trajectory: straight leg then turn
n_steps = 100;
dt = 1.0;  % seconds
sensor_pos = zeros(3, n_steps);
sensor_vel = zeros(3, n_steps);

% Initial sensor state
pos = [0; 0; 0];
vel = [30; 10; 0];  % m/s

for k = 1:n_steps
    sensor_pos(:, k) = pos;
    sensor_vel(:, k) = vel;

    % Manoeuvre: turn after step 40
    if k > 40
        turn_rate = 0.02;  % rad/s
        speed = norm(vel(1:2));
        heading = atan2(vel(2), vel(1)) + turn_rate * dt;
        vel = [speed * cos(heading); speed * sin(heading); 0];
    end
    pos = pos + vel * dt;
end

% Measurement noise
sigma_az = 1e-3;  % rad (~0.06 deg)
sigma_el = 1e-3;
R = diag([sigma_az^2, sigma_el^2]);

%% Generate measurements
measurements = zeros(n_steps, 2);  % [az, el]
for k = 1:n_steps
    rel = target_pos - sensor_pos(:, k);
    [az, el, ~] = aot.cartesian_to_spherical(rel);
    measurements(k, :) = [az + sigma_az * randn, ...
                          el + sigma_el * randn];
end

%% Initialize filter from first detection
det.azimuth = measurements(1, 1);
det.elevation = measurements(1, 2);
det.noise = R;
det.time = 0;
det.sensor_id = 1;

ekf = aot.initcvmscekf(det, 0.005, 0.05);  % initial 1/r guess
ekf.set_store_history(true);

%% Run tracking loop
state_history = zeros(n_steps, 6);
range_history = zeros(n_steps, 1);
range_std = zeros(n_steps, 1);
true_range = zeros(n_steps, 1);

for k = 1:n_steps
    ekf.predict(dt);

    z = measurements(k, :)';
    ekf.correct(z, R);

    x = ekf.state();
    P = ekf.covariance();

    state_history(k, :) = x';
    range_history(k) = 1.0 / x(3);         % estimated range
    range_std(k) = sqrt(P(3,3)) / x(3)^2;  % approximate range std
    true_range(k) = norm(target_pos - sensor_pos(:, k));
end

%% RTS Smoothing
smoothed = ekf.smooth();
smoothed_range = 1.0 ./ smoothed(:, 3);

%% Plot results
figure('Name', 'Passive Ranging Demo', 'Position', [100 100 1000 600]);

% Range convergence
subplot(2, 2, 1);
t = (1:n_steps) * dt;
plot(t, true_range, 'k-', 'LineWidth', 1.5, 'DisplayName', 'True range');
hold on;
plot(t, range_history, 'b-', 'LineWidth', 1, 'DisplayName', 'EKF estimate');
plot(t, smoothed_range, 'r--', 'LineWidth', 1, 'DisplayName', 'Smoothed');
fill([t, fliplr(t)], ...
     [range_history' + 2*range_std', fliplr(range_history' - 2*range_std')], ...
     'b', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'DisplayName', '2\sigma');
xlabel('Time (s)');
ylabel('Range (m)');
title('Range Convergence');
legend('Location', 'best');
grid on;

% Azimuth tracking
subplot(2, 2, 2);
plot(t, measurements(:,1), 'k.', 'MarkerSize', 3, 'DisplayName', 'Measurements');
hold on;
plot(t, state_history(:,1), 'b-', 'LineWidth', 1, 'DisplayName', 'EKF');
xlabel('Time (s)');
ylabel('Azimuth (rad)');
title('Azimuth Tracking');
legend('Location', 'best');
grid on;

% Elevation tracking
subplot(2, 2, 3);
plot(t, measurements(:,2), 'k.', 'MarkerSize', 3, 'DisplayName', 'Measurements');
hold on;
plot(t, state_history(:,2), 'b-', 'LineWidth', 1, 'DisplayName', 'EKF');
xlabel('Time (s)');
ylabel('Elevation (rad)');
title('Elevation Tracking');
legend('Location', 'best');
grid on;

% Geometry (top-down view)
subplot(2, 2, 4);
plot(sensor_pos(1,:), sensor_pos(2,:), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Sensor path');
hold on;
plot(target_pos(1), target_pos(2), 'r*', 'MarkerSize', 12, 'DisplayName', 'Target');
% Plot a few LOS lines
for k = [1, 25, 50, 75, 100]
    dir = aot.az_el_to_unit_vector(measurements(k,1), measurements(k,2));
    endpoint = sensor_pos(:,k) + dir * true_range(k) * 1.2;
    plot([sensor_pos(1,k), endpoint(1)], [sensor_pos(2,k), endpoint(2)], ...
         'g--', 'LineWidth', 0.5, 'HandleVisibility', 'off');
end
xlabel('X (m)');
ylabel('Y (m)');
title('Geometry (Top-Down)');
legend('Location', 'best');
axis equal;
grid on;

sgtitle('angle\_only: Passive Ranging Demo (MEX)');

fprintf('\n=== Results ===\n');
fprintf('Final true range:      %.1f m\n', true_range(end));
fprintf('Final estimated range: %.1f m\n', range_history(end));
fprintf('Final smoothed range:  %.1f m\n', smoothed_range(end));
fprintf('Final range error:     %.1f m (%.1f%%)\n', ...
    abs(range_history(end) - true_range(end)), ...
    100 * abs(range_history(end) - true_range(end)) / true_range(end));
