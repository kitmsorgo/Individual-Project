function T_anal = T_mixed_flux_exact(x, t, k, rho_c, q0, T0, L, Nmodes)
% T_MIXED_FLUX_EXACT
% Analytical solution for:
%   T_t = alpha T_xx, 0<x<L
%   T(0,t) = T0
%   -k T_x(L,t) = q0 (constant)
%   T(x,0) = T0
%
% x: [1 x Nx], t: [1 x Nt]
% Np: number of modes in the series (Nmodes)

    alpha = k / rho_c;
    x  = x(:);               % column vector
    Nx = numel(x);
    Nt = numel(t);

    n      = 0:Nmodes-1;                     % mode indices
    lambda = (pi/2 + n*pi) / L;              % [1 x N]
    lambda2 = lambda.^2;

    % Coefficients b_n
    bn = (2*q0/(k*L)) * ((-1).^n ./ lambda2);   % [1 x N]

    % Precompute sin(lambda_n x) matrix: [Nx x N]
    S = sin(x * lambda);   % x (Nx x 1) * lambda (1 x N) -> (Nx x N)

    T_anal = zeros(Nx, Nt);

    for it = 1:Nt
        tau = t(it);
        if tau == 0
            T_anal(:, it) = T0;
        else
            decay = exp(-alpha * lambda2 * tau);    % [1 x N]
            v_xt  = S * ( (bn .* decay).' );        % [Nx x 1]
            T_anal(:, it) = T0 - (q0/k)*x + v_xt;
        end
    end
end
