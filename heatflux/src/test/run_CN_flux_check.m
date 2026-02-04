function run_CN_flux_check()
% RUN_CN_FLUX_CHECK
% Sanity check for heat1D_forward_flux:
% Verifies that the numerical right-boundary flux implied by the
% CN temperature field matches the imposed q_right(t).

    %% --------------------------------------------------------------------
    % 1. Define physical / numerical parameters for a simple test case
    %% --------------------------------------------------------------------
    k      = 200;       % W/(m·K)  (example: steel-like)
    rho_c  = 3.9e6;     % J/(m^3·K) (rho * cp, e.g. 7800 * 500)
    L      = 0.10;      % m        (bar length)
    Nx     = 51;        % spatial nodes
    dt     = 0.05;      % s        (time step)
    t_end  = 2.0;       % s        (final time)

    % Constant left boundary and initial temperature
    T_left_val = 300;   % K

    T_init_func  = @(x) T_left_val + 0*x;
    T_left_func  = @(t) T_left_val + 0*t;

    % Imposed constant right-boundary flux (Neumann BC)
    q0           = 5e3; % W/m^2
    q_right_func = @(t) q0 + 0*t;

    %% --------------------------------------------------------------------
    % 2. Run the Crank–Nicolson solver
    %% --------------------------------------------------------------------
    [T, q_left_imp, q_right_imp, x, t] = heat1D_forward_flux( ...
        k, rho_c, L, Nx, dt, t_end, ...
        T_init_func, T_left_func, q_right_func);

    Nt = numel(t);
    dx = x(2) - x(1);

    %% --------------------------------------------------------------------
    % 3. Compute numerical right-boundary flux from temperature field
    %
    %    q_right_num(t_n) = -k * (T_N^n - T_{N-1}^n) / dx
    %% --------------------------------------------------------------------
    q_right_num = -k * (T(end, :) - T(end-1, :)) / dx;   % 1 x Nt

    % Imposed flux from the BC for all times
    q_right_true = arrayfun(q_right_func, t);            % 1 x Nt

    % Error over time
    q_err = q_right_num - q_right_true;

    max_abs_err = max(abs(q_err));
    rms_err     = sqrt(mean(q_err.^2));

    %% --------------------------------------------------------------------
    % 4. Print diagnostics
    %% --------------------------------------------------------------------
    fprintf('========== CN RIGHT-BOUNDARY FLUX CHECK ==========\n');
    fprintf('k        = %.3f W/(m·K)\n', k);
    fprintf('L        = %.3f m\n', L);
    fprintf('Nx       = %d\n', Nx);
    fprintf('dt       = %.4f s\n', dt);
    fprintf('t_end    = %.4f s\n', t_end);
    fprintf('q0 (imposed) = %.3f W/m^2\n\n', q0);

    fprintf('Max |q_num - q_true| over time = %.6e W/m^2\n', max_abs_err);
    fprintf('RMS  error over time          = %.6e W/m^2\n\n', rms_err);

    % Also print values at final time
    fprintf('At t = t_end = %.4f s:\n', t_end);
    fprintf('    q_right_num (from T) = %.6f W/m^2\n', q_right_num(end));
    fprintf('    q_right_true (imposed) = %.6f W/m^2\n', q_right_true(end));
    fprintf('    Error                  = %.6e W/m^2\n', ...
            q_right_num(end) - q_right_true(end));

    %% --------------------------------------------------------------------
    % 5. Plot comparison over time
    %% --------------------------------------------------------------------
    figure;
    subplot(2,1,1);
    plot(t, q_right_true, 'k--', 'LineWidth', 1.0); hold on;
    plot(t, q_right_num,  'o-',  'LineWidth', 1.0);
    xlabel('t [s]');
    ylabel('q_{right} [W/m^2]');
    legend('Imposed q_{right}(t)', 'Numerical q_{right}(t)', ...
           'Location', 'best');
    title('Right-boundary heat flux: imposed vs numerical (CN)');
    grid on;

    subplot(2,1,2);
    plot(t, q_err, 'o-');
    xlabel('t [s]');
    ylabel('q_{num} - q_{true} [W/m^2]');
    title('Flux error at right boundary over time');
    grid on;

end
