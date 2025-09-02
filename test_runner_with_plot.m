
clear;
clc;
close all;	  
%% Parameters

F = 8;			% forcing parameter
he = 1e-3;  		% time step, standard Euler
t_final = 4;        %30   % duration of the simulation in natural time units
NTe = fix(t_final/he);  % no. of discrete time steps, standard Euler

sz = sqrt(1/4); %sqrt(1/4)        % std of the observations; full filtering
sx = sqrt(1/2);         %5  ,sqrt(1/2)      % std of the signal noise / diffusion coefficient
tobs = 0.1;            % continuous time between observations
     	           
% Particle filter parameters

ness_thr = 0.7;		% NESS threshold for resampling
%

%
% Simulation parameters
%
n_steps = ceil(5/he);    % number of time step to skip transient solution      
n_obs = ceil(tobs/he);   % number of subintervals between two observations
filtered_solution_indices=1:n_obs:NTe+1; % filtered solution is computed at these index 
coarse_time_mesh=he.*(0:n_obs:NTe); % time mesh corresponding those inex

full_indices=1:NTe+1;    % ground truth signal and predicted solution are computed at these index
fine_time_mesh=he*(0:NTe);
%% barrier paramemters; 
    r_obs=4*sz;
    barrier_params.p=r_obs; % keep constant
    barrier_params.alpha=1; % optimize
    barrier_params.mu=50;% optimize

    barrier_params.k=80;% optimize
%%
  
  
  Dx=2000;
  N=500;
  Dz = fix(3*Dx/5);          % set full observation
  %Dz=Dx;
  fixed_observed_components = randsample(Dx,Dz);
  fixed_observed_components= sort(fixed_observed_components);
 
  
    % Repetition loop
   
        ok = 0;
        while not(ok)
    
        n_steps = ceil(5/he);%-----------------------------------------> up to half of final time ?
        Wx0 = sqrt(he)*randn([Dx n_steps]);% -------------------------> Brownian increment
        [x_ini,~] = exp_euler(rand([Dx 1]),he,F,n_steps,Dx,Wx0,sx);
        idx = randsample( fix(n_steps/2):n_steps, 1 );%------------------> choose an integer between nstep/2 ,nsptep
        x0 = x_ini(:,idx);
        % now we run with the 'regular' initialisation
        Wx = sqrt(he)*randn([Dx NTe]);%----------------------------------> create new increment
        [x,ok] = exp_euler(x0,he,F,NTe,Dx,Wx,sx); %--------------------> if ok =1, x does not  contain nan values?
                                                  % ------------------> x holds 10 000 realizations
                                                                      % of 800 state  
        end %while
        % for normalization.
        Pd_f = mean( sum( x(:,filtered_solution_indices).^2 ) ) ;  % for filtered comparison
        Pd_p = mean( sum( x.^2 ) ) ;                                % for predicted comparison
       
        H0= eye(Dx) + randn([Dx Dx]).*5e-4;
        H0x = H0*x(1:Dx,(n_obs+1):n_obs:NTe+1); % ---------------->now skip the inital time,start from t=h ,take final time 
        ze_full = H0x + sz*randn(size(H0x));
        H=H0(fixed_observed_components,:); %%% CHOOSE SUB MATRİX AS THE OBSERVATION MATRIX
        ze_sparse = ze_full(fixed_observed_components,:); %------> observed values from t=h up to t=T
   

        X0 = x0 + sx*randn([Dx N]);


        %[Yf_barrier, Yp_barrier,resampling_counter_barrier,W_history_barrier] = barrier_with_plot(F,sx,sz,he,NTe,n_obs,ze_sparse,H,X0,ness_thr,barrier_params);
        [Yf_barrier, Yp_barrier,resampling_counter_barrier,W_history_barrier] = girsanov_with_plot_standardizing(F,sx,sz,he,NTe,n_obs,ze_sparse,H,X0,ness_thr,barrier_params);
         MSEf_barrier= mean( sum( (Yf_barrier - x(:,filtered_solution_indices)).^2 ) )/Pd_f;
         MSEp_barrier= mean( sum( (Yp_barrier - x).^2 ) )/Pd_p;
            
       





