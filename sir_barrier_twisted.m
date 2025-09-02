

function [Xf, Xp,resampling_counter,W_history] = sir_barrier_twisted(F,sx,sz,h,NT,n_obs,z,H,X0,ness_thr,barrier_params)

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
% 
% Xf : filtered states
% Xp : predicted states
%
%
rng(10, 'twister');
% recovers the no. of particles (N), 
[Dx, N] = size(X0);

nt=NT/n_obs; % choose NT and obs such that obs| NT

% initialisation
Xf = zeros([Dx nt+1]);      % filtered estimates, at obs
Xp = zeros([Dx NT+1]);      % predicted estimates, at all times  
Xf(:,1) = mean(X0,2);
Xp(:,1) = mean(X0,2);
Xold = X0;                % auxiliary
lw = zeros([1 N]);        % log-weights  
weight = ones([1 N])/N;        % weights  
% noise variance
s2z = sz^2;
s2x=sx^2;
% time steps
sxsqrth=sx*sqrt(h);
%% Barrier hyperparameters.
HT      = H.';                       % transpose once
alpha2  = barrier_params.alpha^2;                   % cache α²
mu=barrier_params.mu;
p=barrier_params.p; % here is r0;
k=barrier_params.k;



%
% --Time loop
%
resampling_counter=0;

W_history   = cell(nt, 1);   % allocate once
% centers are on the obs space
prev_center=H*mean(X0,2);  % Initialize center of 1st hypercube in OBS from initial particle positions.

for obs_idx=1:nt

   obs_z=z(:,obs_idx); %  i eliminated zeros, z(1) gives observed value at t=h 

   C(:,1) = prev_center; %
   C(:,2) = obs_z;   %as Joaquin suggested;
   [ci,ti]=get_centers_of_hypertube(h,n_obs,C);
   
   
   for inner_idx=1:n_obs
        
        Xdrift = l96dxdt(Xold,F,Dx);
        % barrier term is implemented 
        e    = ci(:,inner_idx) - H * Xold;                % d_y × N
        %J = (e' * e) / alpha2;        % scalar
        J = sum(e.^2, 1) ./ alpha2;  % 1xN  
        % Soft-plus derivative ζ = σ(k(J-p))
        zeta = 1 ./ (1 + exp(-k * (J - p)));
         % Barrier gradient  -(2/α²) ζ Hᵀ e
        q     = HT * e; 
        gradL = -(2/alpha2) * q .* zeta; 
        Xdrift=Xdrift-mu*gradL;
        dWx=sxsqrth*randn(Dx,N);
        Xnew=  Xold + h*Xdrift+dWx;
       

        % prediction
        Xold=Xnew;
        Xp(:,(obs_idx-1)*n_obs+inner_idx+1)=Xnew*weight';
       % fprintf("finer_idx  % d, Xp stored at   %d \n", finer_idx_counter,(obs_idx-1)*n_obs+inner_idx+1);
       
   end 
   
        

   % prediction
         %% CHECK HX IS INSIDE THE HYPERCUBE:
      
  
        
        
       % Available observations

       % weights
       
       llk = -(1/(2*s2z)) .* sum(( obs_z - H*Xnew ).^2);  % log-likelihood
       lw = lw+ llk - max(lw+llk);
       
       wu = exp(lw);
       weight = wu ./ sum(wu);
       % estimation
       Xf(:,obs_idx+1) = Xnew*weight';  %%% is it the correct place ?  (obs_idx+1)
       
        W_history{obs_idx} = weight;     %  % Save weights before resampling
       % fprintf("finer_idx  % d, Xp stored at   %d \n", finer_idx_counter,(obs_idx-1)*n_obs+inner_idx+1);     
       % Resampling
       NESS = (1/sum(weight.^2))/N; 
       if NESS<ness_thr
            idx = randsample(1:N, N, true, weight);
            Xnew(:,1:N) = Xnew(:,idx);
            Xold = Xnew;
            weight= ones([1 N])/N;
            lw = zeros([1 N]);
            resampling_counter=resampling_counter+1;
       end %if
       
   %fprintf("--------------\n");
    
   
    prev_center=C(:,2);  % in the next stage, prev_stage should be independent of SDE samples
    Xold=Xnew;
end % time (n)
