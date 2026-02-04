function run_forward_validation()
% RUN_FORWARD_VALIDATION
% Performs sanity checks + convergence testing on the forward solver
% BEFORE generating all dataset cases.

    fprintf('\n========== FORWARD SOLVER VALIDATION ==========\n');

    %% -----------------------------------------------------------
    % 1. Load config & define test case
    %% -----------------------------------------------------------
    C = experiment_config();

    % Use a simple constant flux case
    k      = C.ranges.k(1);
    rho_c  = C.ranges.rho_c(1);
    L      = C.ranges.L(1);
    T_left = C.ranges.T_left(1);

    % Base numerical settings
    Nx_base = C.ranges.Nx(1);
    dt_base = C.time.dt;  
    t_end   = C.ranges.t_end(1);

    % Flux function for testing
    A = 5e3;   % constant flux
    q_fun = @(t) A + 0*t;

    T_init = @(x) T_left + 0*x;
    T_left_fun = @(t) T_left;

    fprintf('\nTesting with constant flux q = %g W/m^2\n', A);


    %% -----------------------------------------------------------
    % 2. Run base case
    %% -----------------------------------------------------------
    [T_base, qL_base, qR_base, x_base, t_base] = heat1D_forward_flux( ...
        k, rho_c, L, Nx_base, dt_base, t_end, ...
        T_init, T_left_fun, q_fun);

    fprintf('Base case completed: Nx=%d, dt=%.4f\n', Nx_base, dt_base);


    %% -----------------------------------------------------------
    % 3. Finer spatial grid (Nx doubled)
    %% -----------------------------------------------------------
    Nx_fine = Nx_base * 2 - 1;

    [T_fineX, ~, ~, x_fineX, ~] = heat1D_forward_flux( ...
        k, rho_c, L, Nx_fine, dt_base, t_end, ...
        T_init, T_left_fun, q_fun);

    fprintf('Fine spatial grid case completed: Nx=%d\n', Nx_fine);


    %% -----------------------------------------------------------
    % 4. Finer temporal grid (dt halved)
    %% -----------------------------------------------------------
    dt_fine = dt_base / 2;

    [T_fineT, ~, ~, x_fineT, t_fineT] = heat1D_forward_flux( ...
        k, rho_c, L, Nx_base, dt_fine, t_end, ...
        T_init, T_left_fun, q_fun);

    fprintf('Fine temporal grid case completed: dt=%.4f\n', dt_fine);


    %% -----------------------------------------------------------
    % 5. Plot sanity checks for base case
    %% -----------------------------------------------------------

    % Temperature field
    figure; 
    imagesc(t_base, x_base, T_base); 
    set(gca,'YDir','normal');
    xlabel('Time [s]'); ylabel('x [m]');
    title('Temperature Field T(x,t) – Base Case'); 
    colorbar;

    % Boundary flux
    figure;
    plot(t_base, qR_base);
    xlabel('t [s]'); ylabel('Imposed flux q_R(t)');
    title('Right Boundary Flux (Base Case)');


    % Temperature at three key points
    mid_idx = round(numel(x_base)/2);
    figure;
    plot(t_base, T_base(1,:), 'LineWidth', 1.5); hold on;
    plot(t_base, T_base(mid_idx,:), 'LineWidth', 1.5);
    plot(t_base, T_base(end,:), 'LineWidth', 1.5);
    xlabel('Time [s]'); ylabel('Temperature [K]');
    legend('x=0','mid','x=L');
    title('Temperature Evolution – Base Case');


    %% -----------------------------------------------------------
    % 6. Convergence comparison (compare final-time profiles)
    %% -----------------------------------------------------------

    % Interpolate fine solutions onto base grid for comparison
    T_fineX_interp = interp1(x_fineX, T_fineX(:,end), x_base);
    T_fineT_interp = interp1(x_base, T_fineT(:,end), x_base);

    figure;
    plot(x_base, T_base(:,end), '-o', 'LineWidth',1.5); hold on;
    plot(x_base, T_fineX_interp, '--', 'LineWidth',1.5);
    plot(x_base, T_fineT_interp, ':', 'LineWidth',1.5);
    xlabel('x [m]'); ylabel('T(x,t_{end})');
    legend('Base','Fine X','Fine dt');
    title('Convergence Check: Final Temperature Profiles');


% Force to column vectors and compute scalar max error
errX = max(abs(T_fineX_interp(:) - T_base(:,end)));
errT = max(abs(T_fineT_interp(:)   - T_base(:,end)));

fprintf('\n----- Convergence Metrics -----\n');
fprintf('Max |FineX - Base| at t_end: %.6f K\n', errX);
fprintf('Max |FineT - Base| at t_end: %.6f K\n', errT);



    %% -----------------------------------------------------------
    % 7. Simple pass/fail rule
    %% -----------------------------------------------------------
    tol = 0.01;   % 0.01 K acceptable error

if all(errX < tol) && all(errT < tol)
    fprintf('\n✔ Forward solver PASSED convergence test.\n');
    fprintf('✔ Safe to generate full dataset.\n\n');
else
    fprintf('\n✖ Forward solver FAILED convergence test.\n');
    fprintf('✖ DO NOT generate dataset. Fix solver first.\n\n');
end


end
