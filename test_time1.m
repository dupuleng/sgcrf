
m = 500;
n = 800;
p = 200;
lambda = 0.05;

randn('seed', 0);
rand('seed', 0);

[Lambda0 Theta0] = gen_params(n, p);
[X Y] = gen_data(Lambda0, Theta0, m, 0.5);

clear params;
params.max_iters = 50;
params.epsilon = 1e-6;
[Theta Lambda h0] = gcrf_newton(cov([Y X]), p, lambda, params); 

clear params;
params.max_iters = 2000;
params.step = 1e-1;
[Lambda Theta h1] = gcrf_ista(X, Y, lambda, params);

clear params;
params.max_iters = 2000;
params.step = 1e-1;
[Lambda Theta h2] = gcrf_fista(X, Y, lambda, params);

clear params;
params.max_iters = 150;
[Lambda Theta h3] = gcrf_owlqn(X, Y, lambda, params);

x = min([h0.objval h1.objval h2.objval h3.objval]);

figure;
semilogy(h0.time, h0.objval-x, 'b', ...
         h1.time, h1.objval-x, 'r', ...
         h2.time, h2.objval-x, 'g', ...
         h3.time, h3.objval-x, 'm', 'LineWidth', 1);
legend('Newton CD', 'ISTA', 'FISTA', 'OWL-QN');
prepare_figure('time1.pdf', [4 3], 'Time (seconds)', '(f - f^*)/f^*')






