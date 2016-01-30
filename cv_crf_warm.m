function e = cv_crf_warm(Xt, Yt, Xs, Ys, lambda, params)
[m p] = size(Yt); n = size(Xt,2);
St = [Yt Xt]'*[Yt Xt]/m;

warm = logspace(0, log10(lambda*1.1), 10);
params0.max_iters = 10;
params0.tol = 10;
Lambda0 = eye(p);
Theta0 = zeros(n,p);
for i=1:length(warm)
  [Lambda0 Theta0] = gcrf_newton(St, warm(i), Lambda0, Theta0, params0);
end

[Lambda Theta] = gcrf_newton(St, lambda, Lambda0, Theta0, params);
Pt  = -Xt*Theta / Lambda;
Ps  = -Xs*Theta / Lambda;

e = [mse(Pt, Yt) mse(Ps, Ys)];
