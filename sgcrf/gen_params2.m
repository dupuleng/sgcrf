function Omega = gen_params2(n,p)

Omega = zeros(n+p);
idx = randsample((n+p)^2, 0.05*(n+p)^2);
Omega(idx) = ones(numel(idx),1);
Omega = Omega + Omega';
Omega = Omega - diag(diag(Omega));
v = eig(Omega);
sigma = (max(v) - (n+p)*min(v)) / (n+p-1);
Omega = Omega + sigma*eye(n+p);
Omega(p+1:p+n, p+1:p+n) = Omega(p+1:p+n, p+1:p+n) + ones(n,n);



