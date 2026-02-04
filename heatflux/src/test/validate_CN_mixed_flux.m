function validate_CN_mixed_flux()
% VALIDATE_CN_MIXED_FLUX
% Strong verification of heat1D_forward_flux against the
% exact series solution for:
%   T(0,t)=T0,  -k T_x(L,t)=q0,  T(x,0)=T0.

    %% 1. Physical parameters
    k      = 200;        % W/(m·K)
    rho_c  = 3.9e6;      % J/(m^3·K)
    T0     = 300;        % K
    q0     = 5e3;        % W/m^2

    L      = 0.20;       % m (any reasonable finite length)
    t_end  = 2.0;        % s

    %% 2. Grids for convergence study
    grids = struct( ...
        'name', {'Base', 'Fine-X', 'Fine-T'}, ...
        'Nx',   {51,     101,      51}, ...
        'dt',   {0.05,   0.05,     0.025} );

    Nmodes = 200;        % number of modes in analytical series

    for g = 1:numel(grids)
        Nx = grids(g).Nx;
        dt = grids(g).dt;

        Nt = floor(t_end / dt) + 1;
        t  = linspace(0, t_end, Nt);
        x  = linspace(0, L, Nx);

        % BC/IC functions for solver
        T_init_func  = @(x) T0 + 0*x;
        T_left_func  = @(t) T0 + 0*t;
        q_right_func = @(t) q0 + 0*t;

        % Run CN solver
        [T_CN, ~, ~, x_CN, t_CN] = heat1D_forward_flux( ...
            k, rho_c, L, Nx, dt, t_end, ...
            T_init_func, T_left_func, q_right_func);

        % Analytical solution on same grid
        T_anal = T_mixed_flux_exact(x_CN, t_CN, k, rho_c, q0, T0, L, Nmodes);

        % Error at final time
        err_final = T_CN(:, end) - T_anal(:, end);
        max_err   = max(abs(err_final));
        rms_err   = sqrt(mean(err_final.^2));

        fprintf('\n---- %s grid ----\n', grids(g).name);
        fprintf('Nx = %d, dt = %.4f s\n', Nx, dt);
        fprintf('Max error at t_end: %.6e K\n', max_err);
        fprintf('RMS error at t_end: %.6e K\n', rms_err);

        % Optional: whole (x,t) error
        err_all = T_CN - T_anal;
        max_err_all = max(abs(err_all(:)));
        rms_err_all = sqrt(mean(err_all(:).^2));
        fprintf('Max error over all (x,t): %.6e K\n', max_err_all);
        fprintf('RMS error over all (x,t): %.6e K\n', rms_err_all);

        % Plot for base grid
        if g == 1
            figure;
            plot(x_CN, T_anal(:,end), 'k--', 'LineWidth', 1.0); hold on;
            plot(x_CN, T_CN(:,end), 'o-', 'LineWidth', 1.0);
            xlabel('x [m]');
            ylabel('T(x,t_{end}) [K]');
            legend('Analytical series', 'Crank–Nicolson', 'Location','best');
            title('CN vs analytical mixed-BC solution at t_{end}');
            grid on;
        end
    end

    fprintf('\nIf the solver is correct, errors should be small and\n');
    fprintf('should decrease when you refine Nx and/or dt (roughly\n');
    fprintf('second-order in both space and time).\n');

end
