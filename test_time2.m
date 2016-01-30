
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

clear params;
params.max_iters = 600;
params.step = 1e-1;
[Lambda Theta h1] = gcrf_fista(S, p, lambda, params);

clear params;
params.max_iters = 200;
[Lambda Theta h2] = gcrf_owlqn(S, p, lambda, params);

figure;
x = min([h0.objval h1.objval h2.objval]);
semilogy(h0.time/60, (h0.objval-x)/x, 'b', ...
         h1.time/60, (h2.objval-x)/x, 'r', ...
         h2.time/60, (h3.objval-x)/x, 'g', 'LineWidth', 1);
xlim([0 120])
legend('Newton CD', 'FISTA', 'OWL-QN');
prepare_figure('time2.pdf', [4 3], 'Time (minutes)', '(f - f^*)/f^*')

save('test_time2', 'h0', 'h1', 'h2');





