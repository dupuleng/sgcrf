function f = newton_obj(Lambda, Theta, U, V, S, Sigma)
[n p] = size(Theta);
Syy = S(1:p,1:p);
Sxy = S(p+1:end, 1:p);
Syx = S(1:p, p+1:end);
Sxx = S(p+1:end, p+1:end);

d1 = -trace(Sigma*U) + trace(Syy*U) + 2*trace(Syx*V) + ...
     2*trace(Sigma*Theta'*Sxx*V) - trace(Sigma*Theta'*Sxx*Theta*Sigma*U);
d2 = trace(Sigma*U*Sigma*U) + 2*trace(Sigma*V'*Sxx*V) ...
     -4*trace(Sigma*U*Sigma*Theta'*Sxx*V) ...
     +2*trace(Sigma*U*Sigma*Theta'*Sxx*Theta*Sigma*U);
f = d1 + 0.5*d2;
