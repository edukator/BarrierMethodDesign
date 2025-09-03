clear;
clc;
close all;	  
%% Parameters

F = 8;                 % forcing parameter
he = 1e-3;             % time step, standard Euler
t_final = 10;           % duration of the simulation in natural time units
NTe = fix(t_final/he); % no. of discrete time steps, standard Euler

sz = 1;        % std of the observations; full filtering
sx = 1;        % std of the signal noise / diffusion coefficient
tobs = 0.1;            % continuous time between observations

% Particle filter parameters
ness_thr = 0.7;        % NESS threshold for resampling

% Simulation parameters
n_steps = ceil(5/he);                  % number of time step to skip transient solution      
n_obs   = ceil(tobs/he);               % number of subintervals between two observations
filtered_solution_indices = 1:n_obs:NTe+1;       % filtered solution is computed at these indices
coarse_time_mesh = he.*(0:n_obs:NTe);            % time mesh at observation times
full_indices = 1:NTe+1;                           % ground truth & predicted at these indices
fine_time_mesh = he*(0:NTe);

%% Barrier parameters
r_obs = 4*sz;
barrier_params.p     = r_obs; % keep constant
barrier_params.alpha = 1;     % optimize
barrier_params.mu    = 6;     % optimize
barrier_params.k     = 4;     % optimize

%% Dimensions, particles, and observation pattern
Dx = 750;
N  = 500;
Dz = fix(3*Dx/5);                    % partial observation
fixed_observed_components = randsample(Dx, Dz);
fixed_observed_components = sort(fixed_observed_components);

%% Generate trajectory and observations
ok = 0;
while ~ok
    n_steps = ceil(5/he);
    Wx0 = sqrt(he)*randn([Dx n_steps]);                 % Brownian increments
    [x_ini,~] = exp_euler(rand([Dx 1]),he,F,n_steps,Dx,Wx0,sx);
    idx = randsample(fix(n_steps/2):n_steps, 1);
    x0 = x_ini(:,idx);

    Wx = sqrt(he)*randn([Dx NTe]);                      % fresh increments
    [x,ok] = exp_euler(x0,he,F,NTe,Dx,Wx,sx);           % ok==1 if no NaNs
end

% For normalization
Pd_f = mean( sum( x(:,filtered_solution_indices).^2 ) );  % filtered comparison
Pd_p = mean( sum( x.^2 ) );                                % predicted comparison

% Observation model
H0  = eye(Dx) + randn([Dx Dx]).*5e-4;
H0x = H0 * x(1:Dx,(n_obs+1):n_obs:NTe+1);    % from t=h to T (observation times)
ze_full = H0x + sz*randn(size(H0x));         % synthetic full-Dx observations (not all used)
H   = H0(fixed_observed_components,:);       % sensed rows
ze_sparse = ze_full(fixed_observed_components,:);

% Initial ensemble
X0 = x0 + sx*randn([Dx N]);

%% Run filters (your functions are assumed available on path)
[Yf_barrier,   Yp_barrier,  rc_barrier,  W_hist_barrier ] = sir_barrier(F,sx,sz,he,NTe,n_obs,ze_sparse,H,X0,ness_thr,barrier_params);
MSEf_barrier = mean( sum( (Yf_barrier - x(:,filtered_solution_indices)).^2 ) )/Pd_f;
MSEp_barrier = mean( sum( (Yp_barrier - x).^2 ) )/Pd_p;

[Yf_normalize, Yp_normalize, rc_norm, W_hist_norm] = girsanov_with_plot_normalizing(F,sx,sz,he,NTe,n_obs,ze_sparse,H,X0,ness_thr,barrier_params);
MSEf_normalize = mean( sum( (Yf_normalize - x(:,filtered_solution_indices)).^2 ) )/Pd_f;
MSEp_normalize = mean( sum( (Yp_normalize - x).^2 ) )/Pd_p;

[Yf_scale,     Yp_scale,     rc_scale, W_hist_scale] = girsanov_with_plot_variance_scale(F,sx,sz,he,NTe,n_obs,ze_sparse,H,X0,ness_thr,barrier_params);
MSEf_scale = mean( sum( (Yf_scale - x(:,filtered_solution_indices)).^2 ) )/Pd_f;
MSEp_scale = mean( sum( (Yp_scale - x).^2 ) )/Pd_p;

[Yf_std,       Yp_std,       rc_std,   W_hist_std] = girsanov_with_plot_standardizing(F,sx,sz,he,NTe,n_obs,ze_sparse,H,X0,ness_thr,barrier_params);
MSEf_std= mean( sum( (Yf_std - x(:,filtered_solution_indices)).^2 ) )/Pd_f;
MSEp_std = mean( sum( (Yp_std - x).^2 ) )/Pd_p;

[Yf_enkf, Yp_enkf] = enkfH_e(F,sx,sz,he,NTe,n_obs,ze_sparse,H,X0,Dz,fixed_observed_components);
MSEf_enkf = mean( sum( (Yf_enkf - x(:,filtered_solution_indices)).^2 ) )/Pd_f;
MSEp_enkf = mean( sum( (Yp_enkf - x).^2 ) )/Pd_p;

%% === CALL THE SINGLE PLOTTING ROUTINE TWICE =============================

% 1) Plot an OBSERVED component
obs_comp = fixed_observed_components( min(10, numel(fixed_observed_components)) );
plot_component_timeseries( ...
    obs_comp, fixed_observed_components, ...
    fine_time_mesh, coarse_time_mesh, n_obs, NTe, ...
    x, ze_full, ...
    Yp_barrier,   Yf_barrier, ...
    Yp_normalize, Yf_normalize, ...
    Yp_scale,     Yf_scale, ...
    Yp_enkf,      Yf_enkf, ...
    'Observed component' );

% 2) Plot a NON-OBSERVED component
non_observed_components = setdiff(1:Dx, fixed_observed_components);
unobs_comp = non_observed_components(1);
plot_component_timeseries( ...
    unobs_comp, fixed_observed_components, ...
    fine_time_mesh, coarse_time_mesh, n_obs, NTe, ...
    x, ze_full, ...
    Yp_barrier,   Yf_barrier, ...
    Yp_normalize, Yf_normalize, ...
    Yp_scale,     Yf_scale, ...
    Yp_enkf,      Yf_enkf, ...
    'Non-observed component' );

%% ========================= Helper plotting routine ======================
function plot_component_timeseries(comp_idx, observed_idx, ...
    fine_t, coarse_t, n_obs, NTe, ...
    x, ze_full, ...
    Yp_barrier, Yf_barrier, ...
    Yp_norm, Yf_norm, ...
    Yp_scale, Yf_scale, ...
    Yp_enkf, Yf_enkf, ...
    fig_title)

    is_observed = ismember(comp_idx, observed_idx);

    figure('Color','w'); 
    tl = tiledlayout(4,1,'TileSpacing','compact','Padding','compact');
    title(tl, sprintf('%s (i = %d)', fig_title, comp_idx), 'Interpreter','none');

    % nested panel helper
    function plot_panel(ax, method_title, Yp, Yf)
        axes(ax); 
        plot(fine_t, x(comp_idx, :), 'k-', 'LineWidth', 1); hold on;

        if is_observed
            % observations exist only at observation times t = coarse_t(2:end)
            plot(coarse_t(2:end), ze_full(comp_idx, :), 'co', 'LineWidth', 1);
            has_obs = true;
        else
            has_obs = false;
        end

        % predicted (particle-smoothed) is defined for t = he*(1:NTe)
        plot(fine_t(2:end), Yp(comp_idx, 1:end-1), 'm.', 'LineWidth', 1);

        % filtered is at observation grid t = coarse_t
        plot(coarse_t, Yf(comp_idx, :), 'gs--', 'LineWidth', 2);

        % legend
        if has_obs
            legend({'true signal $x(t)$','observations','d-Smooth pred. (Xp)','d-Smooth updt (Xf)'}, ...
                   'Interpreter','latex','Location','best');
        else
            legend({'true signal $x(t)$','d-Smooth pred. (Xp)','d-Smooth updt (Xf)'}, ...
                   'Interpreter','latex','Location','best');
        end

        title(method_title, 'Interpreter','none');
        xlabel('time','Interpreter','latex');
        ylabel('$x_i(t)$','Interpreter','latex');
        grid on; hold off;
    end

    % Create 4 panels
    ax1 = nexttile(tl,1); plot_panel(ax1, 'Barrier',            Yp_barrier, Yf_barrier);
    ax2 = nexttile(tl,2); plot_panel(ax2, 'Barrier normalize',  Yp_norm,    Yf_norm);
    ax3 = nexttile(tl,3); plot_panel(ax3, 'Barrier scale',      Yp_scale,   Yf_scale);
    ax4 = nexttile(tl,4); plot_panel(ax4, 'EnKF',               Yp_enkf,    Yf_enkf);
end






