function [Xf, Xp, number_of_acceptance, W_history] = sir_barrier_ar(F, sx, sz, h, NT, n_obs, z, H, X0, ness_thr, r0, barrier_params, plot_weights)
%SIR_BARRIER_AR Barrier-guided proposal with accept/reject filtering.
%   This routine propagates particles under a barrier-guided drift and
%   discards those leaving the observation hypercube.  No Girsanov
%   correction is applied.  Optionally plots weights before resampling.
%
%   F, sx, sz, h, NT, n_obs, z, H, X0, ness_thr, r0, barrier_params :
%       standard SIR/barrier parameters (see documentation).
%   plot_weights : (optional) if true, bar plot weights before resampling.
%
%   Xf, Xp  : filtered and predicted state estimates.
%   number_of_acceptance : number of surviving particles at each obs step.
%   W_history : cell array storing weights prior to resampling.

if nargin < 13
    plot_weights = false;
end

[Dx, N] = size(X0);
nt = NT / n_obs;

% preallocate outputs
Xf = zeros(Dx, nt+1);
Xp = zeros(Dx, NT+1);
Xf(:,1) = mean(X0,2);
Xp(:,1) = mean(X0,2);
number_of_acceptance = zeros(1, nt);
W_history = cell(nt,1);

% state and weight containers
Xold = X0;
weight = ones(1,N) / N;
lw = zeros(1,N);

% constants
s2z = sz^2;
sxsqrth = sx * sqrt(h);
HT = H.';
alpha2 = barrier_params.alpha^2;
mu = barrier_params.mu;
p = barrier_params.p;
k = barrier_params.k;

prev_center = H * mean(X0,2);

for obs_idx = 1:nt
    obs_z = z(:, obs_idx);
    C = [prev_center, obs_z];
    ci = get_centers_of_hypertube(h, n_obs, C);

    % propagate between observations
    for inner_idx = 1:n_obs
        Xdrift = l96dxdt(Xold, F, Dx);
        e = ci(:,inner_idx) - H * Xold;
        J = sum(e.^2,1) ./ alpha2;
        zeta = 1 ./ (1 + exp(-k*(J - p)));
        q = HT * e;
        gradL = -(2/alpha2) * q .* zeta;
        Xdrift = Xdrift - mu * gradL;
        dWx = sxsqrth * randn(Dx, N);
        Xnew = Xold + h * Xdrift + dWx;
        Xold = Xnew;
        Xp(:, (obs_idx-1)*n_obs + inner_idx + 1) = Xnew * weight';
    end

    % accept/reject step
    idx_acc = [];
    X_acc = zeros(Dx,0);
    counter = 0;
    for par_idx = 1:N
        if is_inside_hypercube(H * Xnew(:,par_idx), obs_z, r0)
            counter = counter + 1;
            idx_acc(counter) = par_idx; %#ok<AGROW>
            X_acc(:,counter) = Xnew(:,par_idx); %#ok<AGROW>
        end
    end
    number_of_acceptance(obs_idx) = counter;
    if counter == 0
        fprintf('Barrier AR: all particles died at %d\n', obs_idx);
        Xf(:) = NaN;
        Xp(:) = NaN;
        return
    end

    lw_acc = lw(idx_acc);
    llk = -(1/(2*s2z)) .* sum((obs_z - H * X_acc).^2);
    lw_acc = lw_acc + llk;
    lw_acc = lw_acc - max(lw_acc);
    wu = exp(lw_acc);
    weight = wu / sum(wu);
    Xf(:, obs_idx+1) = X_acc * weight';
    W_history{obs_idx} = weight;
    if plot_weights
        figure; bar(weight); title(sprintf('Weights before resampling at obs %d', obs_idx));
    end

    % resample back to N particles
    idx = randsample(1:counter, N, true, weight);
    Xold = X_acc(:, idx);
    weight = ones(1,N) / N;
    lw = zeros(1,N);
    prev_center = obs_z;
end
end
