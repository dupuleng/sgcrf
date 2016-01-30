
m = 50;
n = 400;
p = 100;

randn('seed', 0);
rand('seed', 0);

clear params;
params.max_iters = 100;
params.tol = 1e-2;

[Lambda0 Theta0] = gen_params(n, p);
lambda = 0.01;

% Tong's method 
[X Y] = gen_data2(m, n, p);
[Lambda Theta h0] = gcrf_newton(cov([Y X]), p, lambda, params);

% Dense normal - 0.1
[X Y] = gen_data(Lambda0, Theta0, m, 0.1);
[Lambda Theta h1] = gcrf_newton(cov([Y X]), p, lambda, params);

% Dense normal - 1
[X Y] = gen_data(Lambda0, Theta0, m, 1);
[Lambda Theta h2] = gcrf_newton(cov([Y X]), p, lambda, params);

% Dense normal - 10
[X Y] = gen_data(Lambda0, Theta0, m, 10);
[Lambda Theta h3] = gcrf_newton(cov([Y X]), p, lambda, params);

figure;
loglog(h0.time, h0.objval-min(h0.objval), 'r', ...
       h1.time, h1.objval-min(h1.objval), 'b', ...
       h2.time, h2.objval-min(h2.objval), 'g', ...
       h3.time, h3.objval-min(h3.objval), 'm', 'LineWidth', 1);
legend('Tong', '\alpha = 0.1', '\alpha = 1', '\alpha = 10');
prepare_figure('probs.pdf', [8 6], 'Time (seconds)', '(f - f^*)/f^*')





