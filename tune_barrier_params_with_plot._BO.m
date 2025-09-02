function results = tune_barrier_params_with_plot()
% TUNE_BARRIER_PARAMS  Bayesian optimization of (mu, k) for sir_barrier
% Alpha is fixed to 1. Produces a 3D plot (mu, k, MSEf) from all valid evaluations.
% Saves plots, table, and best-params info under a subfolder whose name
% includes key parameter values (EXCLUDES: F, X0, ness_thr).

%% ===================== Base configuration ===============================
F = 8;
he = 1e-3;
t_final = 1;
NTe = fix(t_final/he);

sz = sqrt(1/4);
sx = sqrt(1/2);
tobs = 0.05;

ness_thr = 0.7;  

Dx = 2500;
N  = 750;
%Dz = Dx;                          % full observation
Dz = fix(3*Dx/5);
fixed_seed = 42;                  % lock RNG to make objective deterministic

%% ===================== Fixed dataset (deterministic) ====================
dataset = make_fixed_dataset(F,he,t_final,NTe,sz,sx,tobs,ness_thr,Dx,Dz,N,fixed_seed);

%% ===================== Output folder and base filename ==================
% Helper to make number tokens filesystem-friendly
numtok = @(v) strrep(strrep(strrep(sprintf('%.6g', v),'.','p'),'-','m'),'+','p');

% Common part of filenames/foldernames (exclude F, X0, ness_thr)
base_name_common = sprintf(['alpha1', ...
    '_Dx%d_Dz%d_N%d_sz%s_sx%s_he%s_tobs%s_T%s_seed%d'], ...
    Dx, Dz, N, numtok(sz), numtok(sx), numtok(he), numtok(tobs), ...
    numtok(t_final), fixed_seed);

% Subfolder includes param values + timestamp
timestamp = datestr(now,'yyyymmdd_HHMMSS');
outdir_base = fullfile(pwd, 'barrier_bo_results');
outdir = fullfile(outdir_base, [base_name_common, '_', timestamp]);
if ~exist(outdir, 'dir'), mkdir(outdir); end

%% ===================== Objective & variables ============================
% We'll keep a local eval log via a nested function.
eval_log = [];   % rows: [mu, k, MSEf]

objective = @(tbl) obj_wrapper(tbl.mu, tbl.k);

vars = [
    optimizableVariable('mu',[5e-1, 150], 'Transform','log')
    optimizableVariable('k', [5, 100],    'Transform','log')
];

%% ===================== Run Bayesian optimization ========================
BO = bayesopt(objective, vars, ...
    'IsObjectiveDeterministic', true, ...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'MaxObjectiveEvaluations', 100, ...
    'PlotFcn', {@plotMinObjective}, ...
    'Verbose', 1);

best_tbl  = BO.XAtMinObjective;
best_msef = BO.MinObjective;

fprintf('\n=== Best params (BO) ===\n');
fprintf('mu    = %.6g\n', best_tbl.mu);
fprintf('k     = %d\n',    best_tbl.k);
fprintf('MSEf  = %.6g\n',  best_msef);

%% ===================== Build/print table of all evaluations =============
EvalTbl = array2table(eval_log, 'VariableNames', {'mu','k','MSEf'});
% Drop rows with NaN
EvalTbl = EvalTbl(~isnan(EvalTbl.MSEf), :);

% If BO tries identical (mu,k) more than once, keep the best MSEf per pair:
if ~isempty(EvalTbl)
    [~, iu, ~] = unique(EvalTbl(:,{'mu','k'}), 'rows', 'stable');
    EvalTbl = EvalTbl(iu,:);
    EvalTbl = sortrows(EvalTbl, 'MSEf');
end

disp('=== All evaluated points (sorted by MSEf) ===');
disp(EvalTbl);

%% ===================== 3D visualization (mu, k -> MSEf) =================
scatter_fig = [];
surface_fig = [];

if ~isempty(EvalTbl)
    % --- Scatter figure
    scatter_fig = figure('Name','mu-k-MSEf (scatter)','Color','w');
    scatter3(EvalTbl.mu, EvalTbl.k, EvalTbl.MSEf, 20, 'filled');
    grid on; box on;
    xlabel('\mu'); ylabel('k'); zlabel('MSE_f');
    title('All valid evaluations (scatter)');

    % --- Surface figure
    mu_min = min(EvalTbl.mu); mu_max = max(EvalTbl.mu);
    k_min  = min(EvalTbl.k);  k_max  = max(EvalTbl.k);

    ng_mu = 60; ng_k = 60;
    mu_grid = logspace(log10(mu_min), log10(mu_max), ng_mu);
    k_grid  = linspace(k_min, k_max, ng_k);
    [MU, K] = meshgrid(mu_grid, k_grid);

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
end

%% ===================== Saving: plots, table, best params =================
% Build a base name that also includes the BO best (mu,k)
base_name = sprintf('%s_mu%s_k%d', base_name_common, numtok(best_tbl.mu), round(best_tbl.k));

% Save table (if available)
if ~isempty(EvalTbl)
    csv_path = fullfile(outdir, [base_name, '_eval_table.csv']);
    writetable(EvalTbl, csv_path);
end

% Save best params (TXT + MAT)
best_txt = fullfile(outdir, [base_name, '_best_params.txt']);
fid = fopen(best_txt,'w');
fprintf(fid, 'alpha = 1\nmu = %.12g\nk = %d\nMSEf = %.12g\n', best_tbl.mu, round(best_tbl.k), best_msef);
fprintf(fid, 'Dx = %d, Dz = %d, N = %d, sz = %.12g, sx = %.12g, he = %.12g, tobs = %.12g, T = %.12g, seed = %d\n', ...
    Dx, Dz, N, sz, sx, he, tobs, t_final, fixed_seed);
fclose(fid);

save(fullfile(outdir, [base_name, '_best_params.mat']), 'best_tbl', 'best_msef');

% Save plots (PNG + FIG) when available
if ~isempty(scatter_fig) && ishghandle(scatter_fig)
    png1 = fullfile(outdir, [base_name, '_scatter.png']);
    fig1 = fullfile(outdir, [base_name, '_scatter.fig']);
    try
        exportgraphics(scatter_fig, png1, 'Resolution',300);
    catch
        saveas(scatter_fig, png1);
    end
    savefig(scatter_fig, fig1);
end

if ~isempty(surface_fig) && ishghandle(surface_fig)
    png2 = fullfile(outdir, [base_name, '_surface.png']);
    fig2 = fullfile(outdir, [base_name, '_surface.fig']);
    try
        exportgraphics(surface_fig, png2, 'Resolution',300);
    catch
        saveas(surface_fig, png2);
    end
    savefig(surface_fig, fig2);
end

%% ===================== Pack results =====================================
results = struct();
results.best_params = struct('mu', best_tbl.mu, 'k', best_tbl.k, 'alpha', 1);
results.best_msef   = best_msef;
results.BO          = BO;
results.dataset     = dataset;
results.eval_table  = EvalTbl;  % <- mu,k,MSEf table for later use
results.output_dir  = outdir;

%% ===================== Nested objective wrapper =========================
    function f = obj_wrapper(mu, k)
        f = objective_msef(mu, k, dataset, fixed_seed);
        if ~isnan(f)
            eval_log(end+1, :) = [mu, k, f]; %#ok<AGROW>
        end
    end
end

% ========================================================================
function f = objective_msef(mu, k, D, fixed_seed)
% Single deterministic evaluation of MSEf for given (mu,k).
rng(fixed_seed, 'twister');

barrier_params = struct();
barrier_params.p     = D.r_obs;
barrier_params.alpha = 1;         % alpha fixed
barrier_params.mu    = mu;
barrier_params.k     = k;

[Yf, ~, ~, ~] = sir_barrier(D.F, D.sx, D.sz, D.he, D.NTe, D.n_obs, ...
                            D.ze_sparse, D.H, D.X0, D.ness_thr, barrier_params);

if any(isnan(Yf(:)))
    f = NaN;   % mark as invalid
else
    f = mean( sum( (Yf - D.x(:, D.filtered_solution_indices)).^2, 1 ) ) / D.Pd_f;
end
end

% ========================================================================
function D = make_fixed_dataset(F,he,t_final,NTe,sz,sx,tobs,ness_thr,Dx,Dz,N,fixed_seed)
rng(fixed_seed, 'twister');
n_obs = ceil(tobs/he);
filtered_solution_indices = 1:n_obs:NTe+1;

ok = 0;
while ~ok
    n_steps = ceil(5/he);
    Wx0 = sqrt(he) * randn(Dx, n_steps);
    x_rand0 = rand(Dx,1);
    [x_ini,~] = exp_euler(x_rand0, he, F, n_steps, Dx, Wx0, sx);
    idx = randsample(fix(n_steps/2):n_steps, 1);
    x0 = x_ini(:,idx);

    Wx = sqrt(he) * randn(Dx, NTe);
    [x,ok] = exp_euler(x0, he, F, NTe, Dx, Wx, sx);
end

Pd_f = mean( sum( x(:, filtered_solution_indices).^2, 1 ) );
Pd_p = mean( sum( x.^2, 1 ) );

H0  = eye(Dx) + 5e-4*randn(Dx,Dx);
H0x = H0 * x(1:Dx, (n_obs+1):n_obs:NTe+1);
ze_full   = H0x + sz*randn(size(H0x));

fixed_observed_components = sort(randsample(Dx, Dz));
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
