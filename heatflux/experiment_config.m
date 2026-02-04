function C = experiment_config()
% EXPERIMENT_CONFIG
% Central configuration for heat-flux data generation.

    % ---- Parameter ranges (currently scalars, can be expanded later) ----
    C.ranges.L      = 0.01;      % Plate thickness [m]
    C.ranges.k      = 10.0;      % Thermal conductivity [W/mK]
    C.ranges.rho_c  = 3.8e6;     % rho * cp [J/m^3K]
    C.ranges.Nx     = 51;        % Number of spatial nodes
    C.ranges.t_end  = 50.0;      % Final time [s]
    C.ranges.T_left = 293.15;    % Left boundary temperature [K]

    % ---- Time-stepping ----
    C.time.dt = 0.05;            % Time step [s]

    % ---- Paths ----
    this_file = mfilename('fullpath');
    root_dir  = fileparts(this_file);      % assume this is project root or close enough

    C.paths.data_root = fullfile(root_dir, 'data');
    C.paths.data_raw  = fullfile(C.paths.data_root, 'raw');
    C.paths.data_meta = fullfile(C.paths.data_root, 'meta');

    % ---- IO ----
    C.io.manifest_csv = fullfile(C.paths.data_root, 'manifest.csv');
end
