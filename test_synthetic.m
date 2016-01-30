m = 2500;
n = 4000;
p = 1000;
lambda = 0.2;

randn('seed', 0);
rand('seed', 0);

[Lambda0 Theta0] = gen_params3(n, p);
[X Y] = gen_data(Lambda0, Theta0, m);
S = cov([Y X]);

clear params;
params.max_iters = 20;
params.epsilon = 1e-6;
[Lambda Theta h0] = gcrf_newton(S, lambda, eye(p), zeros(n,p), params); 






