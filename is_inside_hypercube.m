function ii=is_inside_hypercube(x,C,r)
% function ii=checkhypercube(x,C,r)
% Check with indices of x lie out of hypercube center C with sides length 2r, so radius is r.
  myeps=1e-14;
  ii=~any((abs(x-C)-r)>myeps);
end
