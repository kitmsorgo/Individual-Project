function cases = define_flux_cases(C)
% DEFINE_FLUX_CASES
% Produce a uniform struct array describing all experiment cases.

    % ---- Uniform template for all cases ----
    template = struct( ...
        'case_id',        "", ...
        'L',              [], ...
        'k',              [], ...
        'rho_c',          [], ...
        'Nx',             [], ...
        't_end',          [], ...
        'T_left',         [], ...
        'q_right_type',   "", ...
        'q_right_params', [] );

    % Prepare list (we will append rows)
    cases = template;     % first element
    idx = 0;

    % Extract scalars from config
    L      = C.ranges.L(1);
    k      = C.ranges.k(1);
    rho_c  = C.ranges.rho_c(1);
    Nx     = C.ranges.Nx(1);
    t_end  = C.ranges.t_end(1);
    T_left = C.ranges.T_left(1);


    %% ----------- 1. CONSTANT FLUX CASES -------------
    A_list = [5e3, 1e4, 1.5e4];
    for A = A_list
        idx = idx + 1;

        cases(idx) = template;          % start from template
        cases(idx).case_id        = sprintf('const_%g', A);
        cases(idx).L              = L;
        cases(idx).k              = k;
        cases(idx).rho_c          = rho_c;
        cases(idx).Nx             = Nx;
        cases(idx).t_end          = t_end;
        cases(idx).T_left         = T_left;
        cases(idx).q_right_type   = 'constant';
        cases(idx).q_right_params = struct('A', A);
    end

    % New constant flux cases (between trained values)
    A_new_list = [6e3, 1.2e4, 1.4e4, 2e4];  % 6000, 12000, 14000, 20000 (edge)
    for A = A_new_list
        idx = idx + 1;

        cases(idx) = template;
        cases(idx).case_id        = sprintf('const_%g', A);
        cases(idx).L              = L;
        cases(idx).k              = k;
        cases(idx).rho_c          = rho_c;
        cases(idx).Nx             = Nx;
        cases(idx).t_end          = t_end;
        cases(idx).T_left         = T_left;
        cases(idx).q_right_type   = 'constant';
        cases(idx).q_right_params = struct('A', A);
    end


    %% ----------- 2. PURE SINE FLUX CASES -------------
    A_list = [5e3, 1e4];
    T_list = [50, 100];      % periods in seconds

    for A = A_list
        for Tper = T_list
            idx = idx + 1;

            omega = 2*pi/Tper;

            cases(idx) = template;
            cases(idx).case_id        = sprintf('sine_A%g_T%g', A, Tper);
            cases(idx).L              = L;
            cases(idx).k              = k;
            cases(idx).rho_c          = rho_c;
            cases(idx).Nx             = Nx;
            cases(idx).t_end          = t_end;
            cases(idx).T_left         = T_left;
            cases(idx).q_right_type   = 'sine';
            cases(idx).q_right_params = struct('A',A,'omega',omega,'phase',0);
        end
    end

    % New sine cases (between trained values and edge cases)
    A_new_list = [7.5e3, 1.25e4];  % 7500, 12500
    T_new_list = [75, 125, 25];   % 75, 125, 25 (edge)
    for A = A_new_list
        for Tper = T_new_list
            idx = idx + 1;

            omega = 2*pi/Tper;

            cases(idx) = template;
            cases(idx).case_id        = sprintf('sine_A%g_T%g', A, Tper);
            cases(idx).L              = L;
            cases(idx).k              = k;
            cases(idx).rho_c          = rho_c;
            cases(idx).Nx             = Nx;
            cases(idx).t_end          = t_end;
            cases(idx).T_left         = T_left;
            cases(idx).q_right_type   = 'sine';
            cases(idx).q_right_params = struct('A',A,'omega',omega,'phase',0);
        end
    end


    %% ----------- 3. OFFSET + SINE CASE -------------
    q0    = 5e3;
    A     = 5e3;
    Tper  = 80;
    omega = 2*pi/Tper;

    idx = idx + 1;

    cases(idx) = template;
    cases(idx).case_id        = 'offset_sine_1';
    cases(idx).L              = L;
    cases(idx).k              = k;
    cases(idx).rho_c          = rho_c;
    cases(idx).Nx             = Nx;
    cases(idx).t_end          = t_end;
    cases(idx).T_left         = T_left;
    cases(idx).q_right_type   = 'offset_sine';
    cases(idx).q_right_params = struct('q0',q0,'A',A,'omega',omega);

    %% ----------- 4. CASES WITH DIFFERENT INITIAL/BOUNDARY TEMPERATURES -------------
    T_left_new_list = [280, 300, 310];  % Different from 293.15
    for T_new = T_left_new_list
        idx = idx + 1;

        cases(idx) = template;
        cases(idx).case_id        = sprintf('const_10000_Tleft_%g', T_new);
        cases(idx).L              = L;
        cases(idx).k              = k;
        cases(idx).rho_c          = rho_c;
        cases(idx).Nx             = Nx;
        cases(idx).t_end          = t_end;
        cases(idx).T_left         = T_new;
        cases(idx).q_right_type   = 'constant';
        cases(idx).q_right_params = struct('A', 1e4);  % Use a trained amplitude
    end

