function results = tune_barrier_ar_params()
%TUNE_BARRIER_AR_PARAMS  Grid search over (mu,k) for barrier AR filter.
%   RESULTS = TUNE_BARRIER_AR_PARAMS() performs a simple grid search over
%   the barrier parameters MU and K for the accept/reject particle filter
%   implemented by SIR_BARRIER_AR.  A fixed dataset is generated so that
%   evaluations are deterministic.  For each valid (mu,k) pair the
%   normalized mean squared error (MSEf) of the filtered estimates is
%   computed.  The function prints the best parameters found and returns a
%   structure summarizing the search.
%
%   The returned structure contains:
%       best_params : struct with fields mu, k, alpha
%       best_msef   : best MSEf value
%       eval_table  : table of all valid evaluations [mu,k,MSEf]
%       dataset     : fixed dataset used in the search
%       mu_list     : vector of mu candidates
%       k_list      : vector of k candidates

%% ===================== Base configuration ===============================
F = 8;
he = 1e-3;
t_final = 8;
NTe = fix(t_final/he);

sz = sqrt(1/4);
sx = sqrt(1/2);
tobs = 0.1;

ness_thr = 0.7;

Dx = 40;                % dimensionality (reduced for speed)
N  = 40;                % number of particles (reduced for speed)
Dz = fix(3*Dx/5);
fixed_seed = 42;        % lock RNG for deterministic dataset

%% ===================== Fixed dataset (deterministic) ====================
D = make_fixed_dataset(F,he,t_final,NTe,sz,sx,tobs,ness_thr,Dx,Dz,N,fixed_seed);

%% ===================== Candidate grids ==================================
mu_list = [5.5, 6, 6.5];
k_list  = [4, 5];

n_mu = numel(mu_list);
n_k  = numel(k_list);

%% ===================== Evaluate grid ====================================
EvalLog = zeros(0,3);  % rows: [mu, k, MSEf]
best_msef = inf;
best_mu = NaN;
best_k  = NaN;

for i = 1:n_mu
    for j = 1:n_k
        mu = mu_list(i);
        k  = k_list(j);
        barrier_params = struct('p', D.r_obs, 'alpha', 1, 'mu', mu, 'k', k);
        [Yf, ~, acc_counts, ~] = sir_barrier_ar(D.F, D.sx, D.sz, D.he, D.NTe, D.n_obs, ...
            D.ze_sparse, D.H, D.X0, D.ness_thr, D.r_obs, barrier_params);
        if any(acc_counts==0) || any(isnan(Yf(:)))
            fprintf('Skipping mu=%.6g, k=%.6g (invalid)\n', mu, k);
            continue;
        end
        MSEf = mean(sum((Yf - D.x(:, D.filtered_solution_indices)).^2,1)) / D.Pd_f;
        EvalLog(end+1,:) = [mu, k, MSEf]; %#ok<AGROW>
        if MSEf < best_msef
            best_msef = MSEf;
            best_mu = mu;
            best_k = k;
        end
    end
end

%% ===================== Report best ======================================
if isempty(EvalLog)
    error('No valid (mu,k) evaluations.');
end
fprintf('Best parameters: mu = %.6g, k = %.6g, MSEf = %.6g\n', best_mu, best_k, best_msef);

%% ===================== Pack results =====================================
results = struct();
results.best_params = struct('mu', best_mu, 'k', best_k, 'alpha', 1);
results.best_msef   = best_msef;
results.eval_table  = array2table(EvalLog, 'VariableNames', {'mu','k','MSEf'});
results.dataset     = D;
results.mu_list     = mu_list;
results.k_list      = k_list;

end

% =========================================================================
function D = make_fixed_dataset(F,he,t_final,NTe,sz,sx,tobs,ness_thr,Dx,Dz,N,fixed_seed)
% Deterministic dataset; avoids dependency on randsample via my_randsample.

rng(fixed_seed, 'twister');
n_obs = ceil(tobs/he);
filtered_solution_indices = 1:n_obs:NTe+1;

ok = 0;
while ~ok
    n_steps = ceil(5/he);
    Wx0 = sqrt(he) * randn(Dx, n_steps);
    x_rand0 = rand(Dx,1);
    [x_ini,~] = exp_euler(x_rand0, he, F, n_steps, Dx, Wx0, sx);
    idx = my_randsample(fix(n_steps/2):n_steps, 1);
    x0 = x_ini(:,idx);

    Wx = sqrt(he) * randn(Dx, NTe);
    [x,ok] = exp_euler(x0, he, F, NTe, Dx, Wx, sx);
end

Pd_f = mean( sum( x(:, filtered_solution_indices).^2, 1 ) );
Pd_p = mean( sum( x.^2, 1 ) );

H0  = eye(Dx) + 5e-4*randn(Dx,Dx);
H0x = H0 * x(1:Dx, (n_obs+1):n_obs:NTe+1);
ze_full   = H0x + sz*randn(size(H0x));

fixed_observed_components = sort(my_randsample(1:Dx, Dz));
H         = H0(fixed_observed_components, :);
ze_sparse = ze_full(fixed_observed_components, :);

rng(fixed_seed+1, 'twister');
X0 = x0 + sx*randn(Dx, N);

D = struct();
D.F = F; D.he = he; D.NTe = NTe;
D.sz = sz; D.sx = sx; D.tobs = tobs; D.alpha = 1;
D.ness_thr = ness_thr;
D.Dx = Dx; D.Dz = Dz; D.N = N;
D.n_obs = n_obs;
D.filtered_solution_indices = filtered_solution_indices;
D.Pd_f = Pd_f; D.Pd_p = Pd_p;
D.H = H; D.ze_sparse = ze_sparse; D.X0 = X0;
D.x = x;
D.r_obs = 4*sz;
end

% =========================================================================
function idx = my_randsample(pool, k)
% Minimal replacement for randsample without Statistics Toolbox.
% pool can be a vector of candidates OR a scalar N meaning 1:N.
if isscalar(pool)
    pool = 1:pool;
end
n = numel(pool);
if k > n
    error('Requested %d samples from %d elements.', k, n);
end
perm = randperm(n, k);
idx  = pool(perm);
end

