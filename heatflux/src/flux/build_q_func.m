function q_fun = build_q_func(type, params)
% BUILD_Q_FUNC
% Construct flux function handle q_fun(t) from type and params.

    switch lower(type)
        case 'constant'
            A = params.A;
            q_fun = @(t) A + 0*t;

        case 'sine'
            A     = params.A;
            omega = params.omega;
            phi   = params.phase;
            q_fun = @(t) A .* sin(omega*t + phi);

        case 'offset_sine'
            q0    = params.q0;
            A     = params.A;
            omega = params.omega;
            q_fun = @(t) q0 + A .* sin(omega*t);

        otherwise
            error('build_q_func:UnknownType', ...
                'Unknown q_right_type: %s', type);
    end
end
