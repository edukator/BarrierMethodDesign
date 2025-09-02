function fx = l96dxdt(x0,F,Dx)
%
% function fx = l961step(x0,F,Dx)
%
% x0 : initial condition
% F : forcing parameter
% Dx : dimension

N = size(x0,2);
fx = zeros([Dx N]);

fx(1,:) = -x0(Dx,:).*(x0(Dx-1,:) - x0(2,:)) - x0(1,:) + F;
fx(2,:) = -x0(1,:).*(x0(Dx,:) - x0(3,:)) - x0(2,:) + F;
for i=3:Dx-1
    fx(i,:) = -x0(i-1,:).*(x0(i-2,:) - x0(i+1,:)) - x0(i,:) + F;
end %i
fx(Dx,:) = -x0(Dx-1,:).*(x0(Dx-2,:) - x0(1,:)) - x0(Dx,:) + F;
    
end

