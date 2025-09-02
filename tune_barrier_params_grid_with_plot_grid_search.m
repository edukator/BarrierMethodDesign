function results = tune_barrier_params_grid_with_plot_grid_search()
% GRID SEARCH for (mu, k) with plotting & saving.
% - Alpha is fixed to 1.
% - Saves plots, table, and best-params info under a subfolder whose name
%   includes key parameter values (EXCLUDES: F, X0, ness_thr).

%% ===================== Base configuration ===============================
F = 8;
he = 1e-3;
t_final = 8;
NTe = fix(t_final/he);

sz = sqrt(1/4);
sx = sqrt(1/2);
tobs = 0.1;

ness_thr = 0.7;

Dx = 750;
N  = 500;
%Dz = Dx;                          % full observation
Dz = fix(3*Dx/5);
fixed_seed = 42;                   % lock RNG to make objective deterministic

%% ===================== Fixed dataset (deterministic) ====================
dataset = make_fixed_dataset(F,he,t_final,NTe,sz,sx,tobs,ness_thr,Dx,Dz,N,fixed_seed);

%% ===================== Output folder and base filename ==================
% Helper to make number tokens filesystem-friendly
numtok = @(v) strrep( strrep( strrep( sprintf('%.6g', v), '.', 'p'), '-', 'm'), '+', 'p');

% Common part of filenames/foldernames (exclude F, X0, ness_thr)
base_name_common = sprintf(['alpha1', ...
    '_Dx%d_Dz%d_N%d_sz%s_sx%s_he%s_tobs%s_T%s_seed%d'], ...
    Dx, Dz, N, numtok(sz), numtok(sx), numtok(he), numtok(tobs), ...
    numtok(t_final), fixed_seed);

% Subfolder includes param values + timestamp
timestamp   = datestr(now,'yyyymmdd_HHMMSS');
outdir_base = fullfile(pwd, 'barrier_finer_grid_results');
outdir      = fullfile(outdir_base, [base_name_common, '_', timestamp]);
if ~exist(outdir, 'dir'), mkdir(outdir); end

%% ===================== GRID of (mu, k) ==================================
% Adjust these lists as desired
%mu_list = [0.5, 1, 5, 10, 25, 50, 75, 100];
%k_list  = [0.5,  5, 25, 50, 75, 100];
mu_list = [3,4, 5, 6,7];
k_list  = [3,4, 5, 6,7];

n_mu = numel(mu_list);
n_k  = numel(k_list);

fprintf('Grid size: %d (mu) x %d (k) = %d evaluations\n', n_mu, n_k, n_mu*n_k);

%% ===================== Evaluate grid (with optional parallel) ===========
use_parallel = true;   % toggle parallelization
EvalLog = NaN(n_mu*n_k, 4);  % [mu, k, MSEf, ResamplingCounter]

pool_started_here = false;
pool_cleanup = []; %#ok<NASGU>  % keep handle alive in scope

if use_parallel
    try
        p = gcp('nocreate');
        if isempty(p)
            % Prefer threads pool if available; fall back to 'local'
            try
                parpool('threads');
            catch
                parpool('local');
            end
            pool_started_here = true;
            % Ensure pool is closed even if errors occur later
            pool_cleanup = onCleanup(@() delete(gcp('nocreate')));
        end
    catch ME
        warning('Could not start a parallel pool (%s); running serially.', ME.message);
        use_parallel = false;
    end
end

if use_parallel
    % Build all pairs
    all_mu = zeros(n_mu*n_k,1);
    all_k  = zeros(n_mu*n_k,1);
    t = 0;
    for i = 1:n_mu
        for j = 1:n_k
            t = t + 1;
            all_mu(t) = mu_list(i);
            all_k(t)  = k_list(j);
        end
    end
    all_msef  = NaN(n_mu*n_k,1);
    all_resam = NaN(n_mu*n_k,1);
    parfor t = 1:(n_mu*n_k)
        [all_msef(t), all_resam(t)] = objective_msef(all_mu(t), all_k(t), dataset, fixed_seed);
    end
    EvalLog(:,1) = all_mu;
    EvalLog(:,2) = all_k;
    EvalLog(:,3) = all_msef;
    EvalLog(:,4) = all_resam;
else
    t = 0;
    tot = n_mu*n_k;
    for i = 1:n_mu
        for j = 1:n_k
            t = t + 1;
            mu = mu_list(i);
            k  = k_list(j);
            [f, rc] = objective_msef(mu, k, dataset, fixed_seed);
            EvalLog(t,:) = [mu, k, f, rc];
            if mod(t, max(1, floor(tot/20)))==0
                fprintf('Progress: %d / %d\n', t, tot);
            end
        end
    end
end

%% ===================== Build/print table of all evaluations =============
EvalTbl = array2table(EvalLog, 'VariableNames', {'mu','k','MSEf','ResamplingCounter'});
% Drop rows with NaN MSEf
EvalTbl = EvalTbl(~isnan(EvalTbl.MSEf), :);

% If identical (mu,k) pairs occurred, keep the best MSEf per pair,
% and the corresponding resampling counter at that min:
if ~isempty(EvalTbl)
    [G, keys] = findgroups(EvalTbl(:,{'mu','k'}));
    msef_min  = splitapply(@min, EvalTbl.MSEf, G);
    resamp_at_min = splitapply(@(msef,resamp) resamp(find(msef==min(msef),1,'first')), ...
                               EvalTbl.MSEf, EvalTbl.ResamplingCounter, G);
    EvalTbl = table(keys.mu, keys.k, msef_min, resamp_at_min, ...
        'VariableNames', {'mu','k','MSEf','ResamplingCounter'});
    EvalTbl = sortrows(EvalTbl, 'MSEf');
end

disp('=== All evaluated points (sorted by MSEf) ===');
topn = min(10, height(EvalTbl));
if topn > 0
    disp(EvalTbl(1:topn,:));
end

%% ===================== Best params ======================================
if isempty(EvalTbl)
    error('No valid evaluations (all NaN). Check your model/evaluator.');
end
best_tbl   = EvalTbl(1,:);
mu_best    = best_tbl{1,'mu'};
k_best     = best_tbl{1,'k'};
best_msef  = best_tbl{1,'MSEf'};

fprintf('\n=== Best params (GRID) ===\n');
fprintf('mu    = %.6g\n', mu_best);
fprintf('k     = %d\n',    round(k_best));
fprintf('MSEf  = %.6g\n',  best_msef);

%% ===================== 3D visualization (mu, k -> MSEf) =================
scatter_fig = [];
surface_fig = [];
heatmap_fig = [];

if ~isempty(EvalTbl)
    % --- Scatter figure
    scatter_fig = figure('Name','mu-k-MSEf (scatter)','Color','w');
    scatter3(EvalTbl.mu, EvalTbl.k, EvalTbl.MSEf, 20, 'filled');
    grid on; box on;
    xlabel('\mu'); ylabel('k'); zlabel('MSE_f');
    title('All valid evaluations (scatter)');

    % --- Surface via scatteredInterpolant (log grid for mu, linear for k)
    mu_min = min(EvalTbl.mu); mu_max = max(EvalTbl.mu);
    k_min  = min(EvalTbl.k);  k_max  = max(EvalTbl.k);

    ng_mu = max(60, n_mu);
    ng_k  = max(60, n_k);
    mu_grid = logspace(log10(mu_min), log10(mu_max), ng_mu);
    k_grid  = linspace(k_min, k_max, ng_k);
    [MU, K] = meshgrid(mu_grid, k_grid);

    can_interp = exist('scatteredInterpolant','class') == 8 || exist('scatteredInterpolant','file') ~= 0;
    if can_interp
        Finterp = scatteredInterpolant(EvalTbl.mu, EvalTbl.k, EvalTbl.MSEf, ...
                                       'natural', 'none');
        Z = Finterp(MU, K);

        surface_fig = figure('Name','mu-k-MSEf (surface)','Color','w');
        surf(MU, K, Z, 'EdgeColor','none');
        hold on;
        plot3(EvalTbl.mu, EvalTbl.k, EvalTbl.MSEf, '.', 'MarkerSize',8);
        hold off;
        grid on; box on; view(135,25);
        xlabel('\mu'); ylabel('k'); zlabel('MSE_f');
        title('Interpolated surface (with samples)');
    else
        warning('scatteredInterpolant not available; skipping surface plot.');
    end

    % --- Heatmap on the exact evaluated grid (robust mapping, no outerjoin)
    [MUu, Ku] = ndgrid(mu_list, k_list);
    M = NaN(size(MUu));   % size = [numel(mu_list)  numel(k_list)]
    for r = 1:height(EvalTbl)
        mu = EvalTbl.mu(r);
        k  = EvalTbl.k(r);
        ii = find(mu_list == mu, 1, 'first');
        jj = find(k_list  == k,  1, 'first');
        if ~isempty(ii) && ~isempty(jj)
            M(ii, jj) = EvalTbl.MSEf(r);
        end
    end

    heatmap_fig = figure('Name','mu-k-MSEf (heatmap)','Color','w');
    imagesc(mu_list, k_list, M'); axis xy;
    colorbar; grid on;
    set(gca,'XScale','log');
    xlabel('\mu'); ylabel('k');
    title('Grid MSE_f (lower is better)');
end

%% ===================== Saving: plots, table, best params =================
base_name = sprintf('%s_mu%s_k%d', base_name_common, numtok(mu_best), round(k_best));

% Save table (now includes ResamplingCounter)
csv_path = fullfile(outdir, [base_name, '_eval_table.csv']);
writetable(EvalTbl, csv_path);

% Save best params (TXT + MAT)
best_txt = fullfile(outdir, [base_name, '_best_params.txt']);
fid = fopen(best_txt,'w');
fprintf(fid, 'alpha = 1\nmu = %.12g\nk = %d\nMSEf = %.12g\n', mu_best, round(k_best), best_msef);
fprintf(fid, 'Dx = %d, Dz = %d, N = %d, sz = %.12g, sx = %.12g, he = %.12g, tobs = %.12g, T = %.12g, seed = %d\n', ...
    Dx, Dz, N, sz, sx, he, tobs, t_final, fixed_seed);
fclose(fid);

save(fullfile(outdir, [base_name, '_best_params.mat']), 'best_tbl', 'best_msef');

% Save plots (PNG + FIG) using helper
save_fig2(scatter_fig, outdir, base_name, 'scatter');
save_fig2(surface_fig, outdir, base_name, 'surface');
save_fig2(heatmap_fig, outdir, base_name, 'heatmap');

%% ===================== Close pool if we opened it =======================
% (onCleanup above already handles it; the following is a no-op if not started here)
if pool_started_here
    delete(gcp('nocreate'));
end

%% ===================== Pack results =====================================
results = struct();
results.best_params = struct('mu', mu_best, 'k', k_best, 'alpha', 1);
results.best_msef   = best_msef;
results.dataset     = dataset;
results.eval_table  = EvalTbl;   % mu,k,MSEf,ResamplingCounter table
results.output_dir  = outdir;
results.mu_list     = mu_list;
results.k_list      = k_list;

end % main function

% ========================================================================
function save_fig2(h, outdir, base_name, tag)
% Helper to save a figure as PNG+FIG, robust across MATLAB versions.
if isempty(h) || ~ishghandle(h), return; end
png_path = fullfile(outdir, [base_name, '_' tag '.png']);
fig_path = fullfile(outdir, [base_name, '_' tag '.fig']);
ok = true;
try
    exportgraphics(h, png_path, 'Resolution',300);
catch
    ok = false;
end
if ~ok
    try, saveas(h, png_path); catch, end
end
try, savefig(h, fig_path); catch, end
end

% ========================================================================
function [f, resampling_counter] = objective_msef(mu, k, D, fixed_seed)
% Single deterministic evaluation of MSEf for given (mu,k).
rng(fixed_seed, 'twister');

barrier_params = struct();
barrier_params.p     = D.r_obs;
barrier_params.alpha = 1;         % alpha fixed
barrier_params.mu    = mu;
barrier_params.k     = k;

[Yf, ~, resampling_counter, ~] = sir_barrier(D.F, D.sx, D.sz, D.he, D.NTe, D.n_obs, ...
                            D.ze_sparse, D.H, D.X0, D.ness_thr, barrier_params);

if any(isnan(Yf(:)))
    f = NaN;   % mark as invalid
else
    f = mean( sum( (Yf - D.x(:, D.filtered_solution_indices)).^2, 1 ) ) / D.Pd_f;
end
end

% ========================================================================
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

% ========================================================================
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
