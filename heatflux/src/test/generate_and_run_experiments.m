function generate_and_run_experiments()
% GENERATE_AND_RUN_EXPERIMENTS
% Drives the entire data-generation pipeline:
%   - load config
%   - make directories
%   - define flux cases
%   - run forward solver for each case
%   - save raw data & metadata
%   - write manifest.csv
    this_file = mfilename('fullpath');
    project_root = fileparts(this_file);
    addpath(genpath(project_root));

    C = experiment_config();
    make_directories(C);

    cases = define_flux_cases(C);
    nCases = numel(cases);

    fprintf('\n=== Generating %d experiment cases ===\n', nCases);

    % Manifest header
    hdr = {'case_id','L','k','rho_c','T_left',...
           'q_right_type','q_right_params',...
           'Nx','t_end','alpha','dx','dt','Nt',...
           'mat_path','meta_path'};

    rows = cell(nCases, numel(hdr));

    dt = C.time.dt;

    for i = 1:nCases
        S = cases(i);

        % Derived values
        alpha = S.k / S.rho_c;
        dx    = S.L / (S.Nx - 1);
        Nt    = floor(S.t_end / dt) + 1;

        % Paths
        mat_path  = fullfile(C.paths.data_raw,  [S.case_id '.mat']);
        meta_path = fullfile(C.paths.data_meta, [S.case_id '_meta.mat']);

        % Initial & boundary conditions
        T_init = @(x) S.T_left + 0*x;
        T_left = @(t) S.T_left;

        % Build flux function
        q_fun = build_q_func(S.q_right_type, S.q_right_params);

        % Run solver
        [T, q_left, q_right, x, time] = heat1D_forward_flux( ...
            S.k, S.rho_c, S.L, S.Nx, dt, S.t_end, ...
            T_init, T_left, q_fun);

        % Save raw data
        save(mat_path, 'T','q_left','q_right','x','time');

        % Save metadata
        meta = S;
        meta.alpha = alpha;
        meta.dx    = dx;
        meta.dt    = dt;
        meta.Nt    = Nt;
        save(meta_path, '-struct', 'meta');

        % Manifest row
        rows(i,:) = {
            S.case_id, S.L, S.k, S.rho_c, S.T_left, ...
            S.q_right_type, jsonencode(S.q_right_params), ...
            S.Nx, S.t_end, alpha, dx, dt, Nt, ...
            mat_path, meta_path};

        fprintf('Completed case %d/%d: %s\n', i, nCases, S.case_id);
    end

    % Write manifest
    cell2csv(C.io.manifest_csv, [hdr; rows]);

    fprintf('\nAll cases generated and solved. Manifest: %s\n', C.io.manifest_csv);
end
