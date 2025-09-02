function [ci,ti]=get_centers_of_hypertube(h,NT,C)
%				%
%  
%
%   h    :time step
%   NT   : number of steps to final time t=hNT  
%   C    : centres of Hypercube at t=0 and t=hNT

%  
%   ci : linearly interpolated centers of hypercube.
%   ti : t(i) = i*h   i=0,1,..,NT
%
%
  
%%%%%%%%%%%%%%%%%%%  
% Linearly interpolate centers over domain of integration.
%  
ti=[0:h:h*NT]';
ci=interp1([0,h*NT]',C',ti);   % transposes needed for interp1
ci=ci';

