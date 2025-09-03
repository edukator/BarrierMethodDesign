function [Xf, Xp, number_of_acceptance, W_history] = sir_barrier_ar_fq(F,sx,sz,h,NT,n_obs,z,H,X0,ness_thr,r0,barrier_params)
%
% Barrier-guided proposal with explicit accept-reject and log f/q correction.
% Only particles whose predicted observation lies inside the hypercube are
% kept and renormalized.  The transition mismatch between the true model
% and the guided proposal is accounted for through the pathwise log f/q term.
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
% number_of_acceptance : number of accepted particles at each observation
%
[Dx, N] = size(X0);
nt = NT/n_obs;

number_of_acceptance = zeros(1,nt);
Xf = zeros([Dx nt+1]);
Xp = zeros([Dx NT+1]);
Xf(:,1) = mean(X0,2);
Xp(:,1) = mean(X0,2);
Xold = X0;
lw = zeros([1 N]);
weight = ones([1 N])/N;

s2z = sz^2;
s2x = sx^2;
sxsqrth = sx*sqrt(h);

HT      = H.';
alpha2  = barrier_params.alpha^2;
mu      = barrier_params.mu;
p       = barrier_params.p;
k       = barrier_params.k;

W_history = cell(nt,1);
prev_center = H*mean(X0,2);

for obs_idx = 1:nt
    obs_z = z(:,obs_idx);
    C(:,1) = prev_center;
    C(:,2) = obs_z;
    [ci, ~] = get_centers_of_hypertube(h,n_obs,C);

    log_correction = zeros(1,N);
    for inner_idx = 1:n_obs
        Xdrift_f = l96dxdt(Xold,F,Dx);
        e    = ci(:,inner_idx) - H * Xold;
        J    = sum(e.^2,1) ./ alpha2;
        zeta = 1 ./ (1 + exp(-k*(J - p)));
        q     = HT * e;
        gradL = -(2/alpha2) * q .* zeta;
        Xdrift_q = Xdrift_f - mu*gradL;

        dWx  = sxsqrth*randn(Dx,N);
        Xnew = Xold + h*Xdrift_q + dWx;
        delta = Xnew - Xold;

        log_correction = log_correction + (1/(2*s2x*h)) * ...
            ( sum((delta - h*Xdrift_q).^2,1) - sum((delta - h*Xdrift_f).^2,1) );

        Xold = Xnew;
        Xp(:,(obs_idx-1)*n_obs + inner_idx + 1) = Xnew*weight';
    end

    % accept-reject step
    X_acc = [];
    idx_acc = [];
    log_corr_acc = [];
    counter = 0;
    for par_idx = 1:N
        if is_inside_hypercube(H*Xnew(:,par_idx), obs_z, r0)
            counter = counter + 1;
            X_acc(:,counter) = Xnew(:,par_idx);
            idx_acc(counter) = par_idx;
            log_corr_acc(counter) = log_correction(par_idx);
        end
    end

    number_of_acceptance(obs_idx) = counter;
    if counter == 0
        fprintf('Barrier AR f/q: all particles died at %d\n', obs_idx);
        return
    end

    lw_acc = lw(idx_acc) + log_corr_acc;
    llk = -(1/(2*s2z)) .* sum( (obs_z - H*X_acc ).^2 );
    lw_acc = lw_acc + llk;
    lw_acc = lw_acc - max(lw_acc);

    wu = exp(lw_acc);
    weight = wu ./ sum(wu);
    Xf(:,obs_idx+1) = X_acc*weight';
    W_history{obs_idx} = weight;

    % resample back to N particles
    idx = randsample(1:counter, N, true, weight);
    X_acc = X_acc(:,idx);
    Xold = X_acc;
    weight = ones([1 N])/N;
    lw = zeros([1 N]);

    prev_center = C(:,2);
end
end
