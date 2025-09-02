function [x,ok] = exp_euler(x0,h,F,NT,Dx,Wx,sx)
%
% function [x,ok] = exp_euler(x0,h,F,NT,Dx,Wx,s2x)
%

x = zeros([Dx NT+1]); %%  NOT  NT, IT IS  NT+1;
x(:,1) = x0;

for n = 2:(NT+1)   %% NOT  NT, IT IS   NT+1;
    driftx = l96dxdt(x(:,n-1),F,Dx);
    x(:,n) = x(:,n-1) + h.*driftx + sx.*Wx(:,n-1);
end %n


   

ok = isempty( find( isnan(x(1,:)) | isinf(x(1,:)), 1 ) );

%fprintf(1,'Explicit Euler: h=%6.5f, Dx=%d, time %6.3f s, ok %d \n', h, Dx, etime(clock,t0), ok);
    

