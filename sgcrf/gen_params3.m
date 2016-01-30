function [Lambda Theta] = gen_params3(n, p, alpha)

Omega = zeros(n+p);
idx = randsample((n+p)^2, 5*(n+p));
Omega(idx) = ones(numel(idx),1);
Omega = Omega + Omega';
Omega = Omega - diag(diag(Omega));
v = eig(Omega);
sigma = (max(v) - (n+p)*min(v)) / (n+p-1);
Omega = Omega + sigma*eye(n+p);

Lambda = Omega(1:p, 1:p);
% slight hack so that L1 applies properly
Theta = 2*Omega(p+1:p+n, 1:p);

