function [Xf, Xp, resampling_counter, W_history] = sir_barrier_indicator_fq(F,sx,sz,h,NT,n_obs,z,H,X0,ness_thr,r0,barrier_params)
%
% Barrier-guided proposal with indicator penalty and log f/q correction.
% At observation times, particles outside the hypercube centered at the
% observation are given zero weight.  The discrepancy between the true
% model and the barrier-guided proposal is corrected via the pathwise
% log-ratio f/q.
%
% F : Lorenz 96 forcing parameter
% sx : process noise std
% sz : observation noise std
% h : Euler integration step
% NT : no. of discrete time steps
% n_obs : observations are collected every n_obs discrete time units
% z : observations
% H : observation matrix
% X0 : initial particles
% ness_thr : resampling threshold
% r0 : hypercube radius for the accept region
% barrier_params : struct with fields alpha, mu, p, k
%
% Xf : filtered states
% Xp : predicted states
% resampling_counter : number of resampling operations
%
% NOTE: this routine has no explicit accept-reject loop.  Particles
% outside the hypercube simply receive zero weight.

[Dx, N] = size(X0);
nt = NT/n_obs; % integer by construction

Xf = zeros([Dx nt+1]);      % filtered estimates, at obs times
Xp = zeros([Dx NT+1]);      % predicted estimates, at all times
Xf(:,1) = mean(X0,2);
Xp(:,1) = mean(X0,2);
Xold = X0;                  % auxiliary
lw = zeros([1 N]);          % log-weights
weight = ones([1 N])/N;     % weights

s2z = sz^2;                 % observation variance
s2x = sx^2;                 % signal variance
sxsqrth = sx*sqrt(h);       % sqrt(h)*sigma

% Barrier hyperparameters
HT      = H.';                       % transpose once
alpha2  = barrier_params.alpha^2;    % cache alpha^2
mu      = barrier_params.mu;
p       = barrier_params.p;
k       = barrier_params.k;

resampling_counter = 0;
W_history = cell(nt,1);
prev_center = H*mean(X0,2);  % center of first hypercube in obs space

for obs_idx = 1:nt
    obs_z = z(:,obs_idx);
    C(:,1) = prev_center;
    C(:,2) = obs_z;
    [ci, ~] = get_centers_of_hypertube(h,n_obs,C);

    log_correction = zeros(1,N);  % pathwise log f/q term
    for inner_idx = 1:n_obs
        % model and proposal drifts
        Xdrift_f = l96dxdt(Xold,F,Dx);
        e    = ci(:,inner_idx) - H * Xold;        % d_y x N
        J    = sum(e.^2, 1) ./ alpha2;
        zeta = 1 ./ (1 + exp(-k * (J - p)));
        q     = HT * e;
        gradL = -(2/alpha2) * q .* zeta;
        Xdrift_q = Xdrift_f - mu*gradL;           % guided drift

        dWx  = sxsqrth*randn(Dx,N);
        Xnew = Xold + h*Xdrift_q + dWx;
        delta = Xnew - Xold;

        % log f/q correction for this step
        log_correction = log_correction + (1/(2*s2x*h)) * ...
            ( sum((delta - h*Xdrift_q).^2,1) - sum((delta - h*Xdrift_f).^2,1) );

        Xold = Xnew;
        Xp(:,(obs_idx-1)*n_obs + inner_idx + 1) = Xnew*weight';
    end

    % weights and indicator
    llk = -(1/(2*s2z)) .* sum( (obs_z - H*Xnew).^2 );
    lw = lw + log_correction + llk;

    inside = false(1,N);
    for par_idx = 1:N
        inside(par_idx) = is_inside_hypercube(H*Xnew(:,par_idx), obs_z, r0);
    end

    if ~any(inside)
        fprintf('Barrier indicator f/q: all particles died at %d\n', obs_idx);
        return
    end

    lw(~inside) = -Inf;  % zero weight outside the region
    maxlw = max(lw(inside));
    lw = lw - maxlw;

    wu = zeros(1,N);
    wu(inside) = exp(lw(inside));
    weight = wu ./ sum(wu);

    Xf(:,obs_idx+1) = Xnew*weight';
    W_history{obs_idx} = weight;

    NESS = (1/sum(weight.^2))/N;
    if NESS < ness_thr
        idx = randsample(1:N, N, true, weight);
        Xnew = Xnew(:,idx);
        Xold = Xnew;
        weight = ones([1 N])/N;
        lw = zeros([1 N]);
        resampling_counter = resampling_counter + 1;
    else
        Xold = Xnew;  % carry forward
    end

    prev_center = C(:,2);
end
end
