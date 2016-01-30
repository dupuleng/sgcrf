[X Y t s] = problem_synthetic_icml();
n = size(X,2);
p = size(Y,2);
m = length(t);

S = [Y(t,:) X(t,:)]'*[Y(t,:) X(t,:)]/m;
lambda = 1;

clear params;
params.max_iters = 100;
params.epsilon = 1e-6;
[Theta Lambda h0] = gcrf_newton(S, lambda, eye(p), zeros(n,p), params); 

clear params;
params.max_iters = 600;
params.step = 1e-2;
[Lambda Theta h1] = gcrf_fista(S, p, lambda, params);

clear params;
params.max_iters = 200;
[Lambda Theta h2] = gcrf_owlqn(S, p, lambda, params);

save('time_synthetic', 'lambda', 'h0', 'h1', 'h2');