classdef GMPHD < handle
    %GMPHD Gaussian Mixture Probability Hypothesis Density filter.
    %   Multi-target tracker that estimates the intensity function (PHD)
    %   as a Gaussian mixture. Handles unknown and time-varying number of targets.
    %
    %   phd = aot.GMPHD()          % default config
    %   phd = aot.GMPHD(config)    % custom config struct
    %
    %   config fields (all optional):
    %     p_survival, p_detection, clutter_rate, merge_threshold,
    %     prune_threshold, max_components, extraction_threshold

    properties (Access = private)
        handle_ uint64
    end

    methods
        function obj = GMPHD(varargin)
            %GMPHD Construct a GM-PHD filter.
            %   phd = aot.GMPHD()       — default parameters
            %   phd = aot.GMPHD(config) — custom config struct
            if nargin == 0
                obj.handle_ = angle_only_mex('gmphd_create');
            else
                obj.handle_ = angle_only_mex('gmphd_create', varargin{1});
            end
        end

        function delete(obj)
            %DELETE Release C++ object.
            if obj.handle_ ~= 0
                angle_only_mex('gmphd_destroy', obj.handle_);
                obj.handle_ = uint64(0);
            end
        end

        function predict(obj, dt)
            %PREDICT Predict all components forward by dt seconds.
            angle_only_mex('gmphd_predict', obj.handle_, dt);
        end

        function correct(obj, measurements)
            %CORRECT Update with measurements (NxM matrix, each row is a measurement).
            angle_only_mex('gmphd_correct', obj.handle_, measurements);
        end

        function merge(obj)
            %MERGE Merge close Gaussian components.
            angle_only_mex('gmphd_merge', obj.handle_);
        end

        function prune(obj)
            %PRUNE Remove low-weight components.
            angle_only_mex('gmphd_prune', obj.handle_);
        end

        function targets = extract(obj)
            %EXTRACT Extract target estimates (components with weight > threshold).
            %   Returns struct array with fields: weight, mean, covariance
            targets = angle_only_mex('gmphd_extract', obj.handle_);
        end

        function n = estimated_target_count(obj)
            %ESTIMATED_TARGET_COUNT Estimated number of targets (sum of weights).
            n = angle_only_mex('gmphd_target_count', obj.handle_);
        end

        function add_birth(obj, births)
            %ADD_BIRTH Add birth components.
            %   births: struct array with fields: weight, mean, covariance
            angle_only_mex('gmphd_add_birth', obj.handle_, births);
        end
    end
end
