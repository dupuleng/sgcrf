function [Lambda1 Theta2] = gcrf_newton_polish(S, lambda, Lambda0, Theta0, params)
[Lambda1 Theta1] = gcrf_newton(S, lambda, Lambda0, Theta0, params);
[n p] = size(Theta0);
Sxx = S(p+1:p+n, p+1:p+n) + 1e-4*eye(n+p);
Sxy = S(p+1:p+n, 1:p);
Theta2 = zeros(n,p);
for i=1:p
  idx = find(Theta1(:,i));
  Theta2(idx,i) = -Sxx(idx,idx) \ Sxy(idx, :)*Lambda1(:,i);
end


