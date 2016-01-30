function g = grad_f2(x, x0, p, S, rho)
if any(isnan(x))
  g = zeros(size(x));
  return;
end

Z = reshape(x, [], p);
Lambda = Z(1:p,:);
Theta = Z(p+1:end,:);

Sigma = inv(Lambda);
Syy = S(1:p,1:p);
Sxy = S(p+1:end,1:p);
Sxx = S(p+1:end,p+1:end);

G = [-Sigma + Syy - Sigma*Theta'*Sxx*Theta*Sigma
     2*Sxy + 2*Sxx*Theta*Sigma];
g = G(:) + rho*(x-x0);
