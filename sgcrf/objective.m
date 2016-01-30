function f = objective(S, lambda, Lambda, Theta, R, Sigma)
[n p] = size(Theta);
Syy = S(1:p, 1:p);
Syx = S(1:p, p+1:p+n);
Sxx = S(p+1:p+n, p+1:p+n);
f = -2*sum(log(diag(R))) + trace(Lambda*Syy) + 2*trace(Theta*Syx) + ...
    trace(Theta*Sigma*Theta'*Sxx) + ...
    lambda*(norm(vec(Theta),1) + ...
            norm(vec(Lambda-diag(diag(Lambda))),1));

