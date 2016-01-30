[X Y] = problem_wind_icml('../data/');
[m n] = size(X); p = size(Y,2);

S = [Y X]'*[Y X]/m;
lambda = 5e-2;

clear params;
params.max_iters = 50;
params.epsilon = 1e-6;
[Theta Lambda h0] = gcrf_newton(S, lambda, eye(p), zeros(n,p), params); 

% clear params;
% params.max_iters = 60;
% params.step = 1e-3;
% [Lambda Theta h1] = gcrf_fista(S, p, lambda, params);

% clear params;
% params.max_iters = 20;
% [Lambda Theta h2] = gcrf_owlqn(S, p, lambda, params);

% save('time_wind', 'lambda', 'h0', 'h1', 'h2');