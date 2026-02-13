classdef MSCEKF < handle
    %MSCEKF Modified Spherical Coordinates Extended Kalman Filter.
    %   Wraps the C++ aot::filters::MSCEKF via MEX.
    %
    %   ekf = aot.MSCEKF()                     % default constructor
    %   ekf = aot.MSCEKF(det, inv_range, std)  % from detection struct
    %   ekf.predict(dt)
    %   ekf.correct(z, R)
    %   x = ekf.state()        % 6x1 [az, el, 1/r, az_dot, el_dot, inv_r_dot]
    %   P = ekf.covariance()   % 6x6

    properties (Access = private)
        handle_ uint64
    end

    methods
        function obj = MSCEKF(varargin)
            %MSCEKF Construct an MSCEKF filter.
            %   ekf = aot.MSCEKF()  — default (zero) state
            %   ekf = aot.MSCEKF(det)  — from detection struct
            %   ekf = aot.MSCEKF(det, inv_range)
            %   ekf = aot.MSCEKF(det, inv_range, inv_range_std)
            %
            %   det is a struct with fields: azimuth, elevation, noise (2x2)
            %   Optional fields: time, sensor_id
            if nargin == 0
                obj.handle_ = angle_only_mex('mscekf_create');
            else
                det = varargin{1};
                args = {det.azimuth, det.elevation, det.noise};
                if isfield(det, 'time'),      args{end+1} = det.time;
                else,                          args{end+1} = 0; end
                if isfield(det, 'sensor_id'),  args{end+1} = det.sensor_id;
                else,                          args{end+1} = 0; end
                if nargin > 1, args{end+1} = varargin{2};  end  % inv_range
                if nargin > 2, args{end+1} = varargin{3};  end  % inv_range_std
                obj.handle_ = angle_only_mex('mscekf_create_from_detection', args{:});
            end
        end

        function delete(obj)
            %DELETE Release C++ object.
            if obj.handle_ ~= 0
                angle_only_mex('mscekf_destroy', obj.handle_);
                obj.handle_ = uint64(0);
            end
        end

        function predict(obj, dt)
            %PREDICT Predict state forward by dt seconds.
            angle_only_mex('mscekf_predict', obj.handle_, dt);
        end

        function correct(obj, z, R)
            %CORRECT Update state with measurement z and noise R.
            %   z: 2x1 [azimuth; elevation]
            %   R: 2x2 measurement noise covariance
            angle_only_mex('mscekf_correct', obj.handle_, z(:), R);
        end

        function correct_jpda(obj, measurements, weights, R)
            %CORRECT_JPDA JPDA-style correction with weighted measurements.
            %   measurements: Nx2 matrix (each row is [az, el])
            %   weights: Nx1 vector of association weights
            %   R: 2x2 measurement noise covariance
            angle_only_mex('mscekf_correct_jpda', obj.handle_, ...
                           measurements, weights(:), R);
        end

        function x = state(obj)
            %STATE Get 6x1 state vector [az, el, 1/r, az_dot, el_dot, inv_r_dot].
            x = angle_only_mex('mscekf_state', obj.handle_);
        end

        function P = covariance(obj)
            %COVARIANCE Get 6x6 state covariance matrix.
            P = angle_only_mex('mscekf_covariance', obj.handle_);
        end

        function d = distance(obj, z, R)
            %DISTANCE Mahalanobis distance to measurement z.
            d = angle_only_mex('mscekf_distance', obj.handle_, z(:), R);
        end

        function L = likelihood(obj, z, R)
            %LIKELIHOOD Gaussian measurement likelihood.
            L = angle_only_mex('mscekf_likelihood', obj.handle_, z(:), R);
        end

        function xs = smooth(obj)
            %SMOOTH RTS fixed-interval smoother over stored history.
            %   Returns Nx6 matrix of smoothed states.
            %   Requires set_store_history(true) before predict/correct.
            xs = angle_only_mex('mscekf_smooth', obj.handle_);
        end

        function set_store_history(obj, flag)
            %SET_STORE_HISTORY Enable/disable history storage for smoothing.
            angle_only_mex('mscekf_set_store_history', obj.handle_, flag);
        end

        function set_state(obj, x)
            %SET_STATE Set the 6x1 state vector.
            angle_only_mex('mscekf_set_state', obj.handle_, x(:));
        end

        function set_covariance(obj, P)
            %SET_COVARIANCE Set the 6x6 covariance matrix.
            angle_only_mex('mscekf_set_covariance', obj.handle_, P);
        end

        function set_process_noise(obj, q)
            %SET_PROCESS_NOISE Set process noise intensity scalar.
            angle_only_mex('mscekf_set_process_noise', obj.handle_, q);
        end
    end
end
