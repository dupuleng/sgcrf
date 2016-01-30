[X Y] = problem_loads_icml('../data/');
[m n] = size(X); p = size(Y,2);

S = [Y X]'*[Y X]/m;
lambda = .05;

clear params;
params.max_iters = 100;
params.epsilon = 1e-6;
[Lambda Theta h0] = gcrf_newton(S, lambda, eye(p), zeros(n,p), params);

clear params;
params.max_iters = 200;
params.step = 1e-2;
[Lambda Theta h1] = gcrf_fista(S, p, lambda, params);

clear params;
params.max_iters = 30;
[Lambda Theta h2] = gcrf_owlqn(S, p, lambda, params);

save('time_loads', 'h0', 'h1', 'h2');