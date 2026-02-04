function [T, q_left, q_right, x, t] = heat1D_forward_flux( ...
        k, rho_c, L, Nx, dt, t_end, ...
        T_init_func, T_left_func, q_right_func)
% HEAT1D_FORWARD_FLUX
% 1D transient heat conduction:
%    T_t = alpha T_xx
% with
%    T(0,t) given (Dirichlet),
%   -k T_x(L,t) = q_right(t) (Neumann flux).
%
% INPUTS:
%   k, rho_c, L, Nx, dt, t_end      – physical / numerical parameters
%   T_init_func(x)                  – initial temperature profile
%   T_left_func(t)                  – left boundary temperature
%   q_right_func(t)                 – imposed heat flux at x=L (W/m^2)
%
% OUTPUTS:
%   T       [Nx x Nt] – temperature field
%   q_left  [1 x Nt]  – computed flux at x=0
%   q_right [1 x Nt]  – imposed flux at x=L
%   x       [1 x Nx]  – spatial grid
%   t       [1 x Nt]  – time grid

    % Grids
    alpha = k / rho_c;
    dx    = L / (Nx - 1);
    x     = linspace(0, L, Nx);
    Nt    = floor(t_end / dt) + 1;
    t     = linspace(0, t_end, Nt);

    % Storage
    T       = zeros(Nx, Nt);
    q_left  = zeros(1, Nt);
    q_right = zeros(1, Nt);

    % Initial condition
    T(:,1) = T_init_func(x);

    % Crank–Nicolson coefficient
    cnoeff = alpha * dt / dx^2;

    % Unknown nodes: i = 2..Nx (Nint = Nx-1)
    Nint = Nx - 1;

    % LHS matrix A
    mainA = (1 + cnoeff) * ones(Nint, 1);
    offA  = (-cnoeff/2) * ones(Nint - 1, 1);
    A = diag(mainA) + diag(offA, 1) + diag(offA, -1);

    % Modify last row for Neumann (flux) BC at x=L
    A(end, :)      = 0;
    A(end, end)    = 1 + cnoeff;
    A(end, end-1)  = -cnoeff;

    % RHS matrix B
    mainB = (1 - cnoeff) * ones(Nint, 1);
    offB  = (cnoeff/2) * ones(Nint - 1, 1);
    B = diag(mainB) + diag(offB, 1) + diag(offB, -1);

    B(end, :)     = 0;
    B(end, end)   = 1 - cnoeff;
    B(end, end-1) = cnoeff;

    % Time stepping
    for n = 1:Nt-1
        tn   = t(n);
        tnp1 = t(n+1);

        % Dirichlet at x=0
        T0n   = T_left_func(tn);
        T0np1 = T_left_func(tnp1);

        T(1,n)   = T0n;
        T(1,n+1) = T0np1;

        % Interior unknowns at time n (nodes 2..Nx)
        Tn_int = T(2:Nx, n);

        % Base RHS
        b = B * Tn_int;

        % Dirichlet contribution at left boundary (node i=2)
        b(1) = b(1) + (cnoeff/2) * (T0n + T0np1);

        % Flux contribution at right boundary
        qn   = q_right_func(tn);
        qnp1 = q_right_func(tnp1);

        % Neumann: -k T_x = q  -> T_x = -q/k
        g_n   = -qn   / k;
        g_np1 = -qnp1 / k;

        flux_term = alpha * dt / dx * (g_n + g_np1);
        b(end) = b(end) + flux_term;

        % Solve linear system for T(2..Nx, n+1)
        T(2:Nx, n+1) = A \ b;

        % Record fluxes
        q_left(n)  = -k * (T(2,n) - T(1,n)) / dx;
        q_right(n) = qn;
    end

    % Final step boundary fluxes
    q_left(Nt)  = -k * (T(2,Nt) - T(1,Nt)) / dx;
    q_right(Nt) = q_right_func(t(end));
end
