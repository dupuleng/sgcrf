function dist_crf_loads(path, n)
lambdas = logspace(0,-2,20);

clear params;
params.max_iters = 1000;
params.tol = 1e-2;

i = str2num(n)+1;
lambda = lambdas(i);

[X Y t s] = problem_loads_icml(path);
n = size(X,2); p = size(Y,2); 

St = [Y(t,:) X(t,:)]'*[Y(t,:) X(t,:)]/length(t);
Lambda0 = eye(p);
Theta0 = zeros(n,p);
[Lambda Theta] = gcrf_newton_polish(St, lambdas(i), Lambda0, Theta0, params);
P = -X*Theta / Lambda;
err = [mse(P(t,:), Y(t,:)) mse(P(s,:), Y(s,:))];

fprintf('CRF %f\t%f\t%f\n', lambda, err);

