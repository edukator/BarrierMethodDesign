function [Xf, Xp, resampling_counter, W_history] = girsanov_with_plot_normalizing( ...
    F, sx, sz, h, NT, n_obs, z, H, X0, ness_thr, barrier_params)


% SIR particle filter for Lorenz-96 with barrier guidance + Girsanov correction.
% Plots per observation:
%   Row 1: single axes with two transparent bar series (meas vs Girsanov).
%   Row 2: bar plot of weights (before any resampling at that obs).
%


% ----------------------- sanity checks ------------------------------------
if mod(NT, n_obs) ~= 0
    error('NT must be divisible by n_obs. Got NT=%d, n_obs=%d.', NT, n_obs);
end
nt = NT / n_obs; % number of observation times

[Dx, N] = size(X0);
Dz = size(H,1);
if size(z,1) ~= Dz || size(z,2) ~= nt
    error('z must be size (Dz x nt) with Dz=%d, nt=%d. Got %dx%d.', Dz, nt, size(z,1), size(z,2));
end

% ----------------------- plotting options ---------------------------------
make_plots   = false;     % set false to disable plotting
save_figs    = false;    % set true to save figures
output_dir   = 'girsanov_vs_meas_figs';
if make_plots && save_figs && ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% ----------------------- allocations --------------------------------------
Xf = zeros(Dx, nt+1);      % filtered estimates, at obs times
Xp = zeros(Dx, NT+1);      % predicted estimates, at fine steps
Xf(:,1) = mean(X0, 2);
Xp(:,1) = mean(X0, 2);

Xold   = X0;                        % current particle cloud (Dx x N)
lw     = zeros(1, N);               % log-weights
weight = ones(1, N) / N;            % normalized weights

s2z = sz^2;                       % observation noise variance
s2x = sx^2;                       % process noise variance
sxsqrth = sx * sqrt(h);           % diffusion increment scale

HT     = H.';                     % transpose once
alpha2 = barrier_params.alpha^2;  % cache α^2
mu     = barrier_params.mu;
p      = barrier_params.p;
k      = barrier_params.k;

resampling_counter = 0;
W_history = cell(nt, 1);

% store (meas_term; girs_term) for each observation


prev_center = H * mean(X0, 2);    % initialize tube center in OBS space from initial mean

% ----------------------- main loops ---------------------------------------
fine_idx_global = 0; % to fill Xp on each fine step

for obs_idx = 1:nt

    obs_z = z(:, obs_idx);        % Dz x 1

    % Two anchors for tube centers in OBS space
    C = zeros(Dz, 2);
    C(:,1) = prev_center;
    C(:,2) = obs_z;

    % get centers for n_obs fine steps
    [ci, ~] = get_centers_of_hypertube(h, n_obs, C); % ci: Dz x n_obs

    deterministic_integral = zeros(1, N);
    stochastic_integral    = zeros(1, N);

    for inner_idx = 1:n_obs
        % Drift of Lorenz-96 for each particle
        Xdrift = l96dxdt(Xold, F, Dx);     % Dx x N

        % Barrier term
        e = ci(:, inner_idx) - H * Xold;                     % Dz x N
        J = sum(e.^2, 1) ./ alpha2;                          % 1 x N
        zeta = 1 ./ (1 + exp(-k * (J - p)));                 % 1 x N
        q = HT * e;                                          % Dx x N
        gradL = -(2/alpha2) * (q .* zeta);                   % Dx x N

        % guided drift
        Xdrift = Xdrift - mu * gradL;

        % diffusion increment
        dWx = sxsqrth * randn(Dx, N);

        % Euler-Maruyama step
        Xnew = Xold + h * Xdrift + dWx;

        % Advance
        Xold = Xnew;

        % Save prediction mean at this fine step
        fine_idx_global = fine_idx_global + 1;
        Xp(:, fine_idx_global + 1) = Xnew * weight.'; % mean wrt current weights

        % --- Girsanov integrals accumulation (1xN each) ---
        deterministic_integral = deterministic_integral + vecnorm(mu * gradL).^2;   % 1xN
        stochastic_integral    = stochastic_integral    + mu * sum(gradL .* dWx, 1);% 1xN
    end

    % finalize Girsanov correction (per particle)
    deterministic_integral = -0.5 / s2x * deterministic_integral * h; % 1xN
    stochastic_integral    = (1 / s2x) * stochastic_integral;         % 1xN
    adjusted_weights_arr   = deterministic_integral + stochastic_integral; % 1xN

    % per-observation terms
    meas_term = -(1/(2*s2z)) .* sum((obs_z - H * Xnew).^2, 1);  % 1xN
    girs_term = adjusted_weights_arr;                            % 1xN

    a_norm = (meas_term - min(meas_term)) ./ ...
         (max(meas_term) - min(meas_term) + eps);
    b_norm = (adjusted_weights_arr - min(adjusted_weights_arr)) ./ ...
         (max(adjusted_weights_arr) - min(adjusted_weights_arr) + eps);

    llk_norm = a_norm + b_norm;

    % --- Compute weights as usual ---
    lw = llk_norm - max(llk_norm);   % subtract max for stability
 

 
    lw = lw - max(lw);           % stability
    wu = exp(lw);
    weight = wu ./ sum(wu);      % <<< weights BEFORE resampling

    % Filtered estimate at this observation time
    Xf(:, obs_idx + 1) = Xnew * weight.';   % (Dx x 1)

    % Save weights pre-resampling
    W_history{obs_idx} = weight;

    meas_term=a_norm;
    girs_term=b_norm;

    % ------------------- 2-ROW PLOT (optional) ----------------------------
    if make_plots
        t_now = obs_idx * n_obs * h;

        % Common y-limits for row 1 (terms)
        y_min = min([meas_term(:); girs_term(:)]);
        y_max = max([meas_term(:); girs_term(:)]);
        y_pad = 0.05 * max(1, y_max - y_min);
        y_lims_terms = [y_min - y_pad, y_max + y_pad];

        % y-limits for row 2 (weights)
        ymax_w = max(weight);
        if ymax_w <= 0, ymax_w = 1; end
        y_lims_w = [0, 1.05 * ymax_w];

        fig = figure('Color','w', 'Name', sprintf('Obs %d (t=%.4g)', obs_idx, t_now));
        tl  = tiledlayout(fig, 2, 1, 'Padding', 'compact', 'TileSpacing', 'compact');

        % -------- Row 1: single axes with two transparent bar series -------
        ax1 = nexttile(tl, 1);
        hold(ax1, 'on');

        b1 = bar(ax1, 1:N, meas_term, 'BarWidth', 1.0, 'DisplayName', ...
                 'Measurement: $-\frac{1}{2\,s_z^{2}}\| z - H X \|^{2}$');
        b1.FaceAlpha = 0.40;  b1.EdgeAlpha = 0.40;

        b2 = bar(ax1, 1:N, girs_term, 'BarWidth', 1.0, 'DisplayName', ...
                 'Adjusted Girsanov');
        b2.FaceAlpha = 0.40;  b2.EdgeAlpha = 0.40;

        grid(ax1, 'on');
        xlim(ax1, [0, N+1]); ylim(ax1, y_lims_terms);
        xlabel(ax1, 'Particle index');
        ylabel(ax1, 'Value');
        title(ax1, sprintf('Obs %d   t = %.4g   (N = %d)', obs_idx, t_now, N), 'FontWeight', 'bold');
        lgd = legend(ax1, 'Location','best'); set(lgd, 'Interpreter','latex');
        yline(ax1, 0, 'k-', 'LineWidth', 0.75);

        % -------- Row 2: bar plot of weights (pre-resampling) --------------
        ax2 = nexttile(tl, 2);
        bar(ax2, 1:N, weight, 'BarWidth', 1.0);
        grid(ax2, 'on');
        xlim(ax2, [0, N+1]); ylim(ax2, y_lims_w);
        xlabel(ax2, 'Particle index');
        ylabel(ax2, 'Weight');
        title(ax2, 'Particle weights (before resampling)');

        sgtitle(tl, sprintf('Observation %d   t = %.4g', obs_idx, t_now), 'FontWeight','bold');

        if save_figs
            fname = fullfile(output_dir, sprintf('obs_%03d_t_%0.6f.png', obs_idx, t_now));
            exportgraphics(fig, fname, 'Resolution', 150);
        end
    end
    % ----------------------------------------------------------------------

    % Resampling by normalized ESS threshold (AFTER plotting weights)
    NESS = (1 / sum(weight.^2)) / N;
    if NESS < ness_thr
        idx = randsample(1:N, N, true, weight); % requires Statistics Toolbox
        Xnew = Xnew(:, idx);
        Xold = Xnew;
        weight = ones(1, N) / N;
        lw = zeros(1, N);
        resampling_counter = resampling_counter + 1;
    end

    % Update prev center for next tube (use the observation anchor)
    prev_center = C(:, 2);

    % Carry Xold forward
    Xold = Xnew;
end

end
