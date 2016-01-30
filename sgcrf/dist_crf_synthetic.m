function dist_crf_synthetic(i)
lambdas = logspace(1, -3, 10);

clear params;
params.max_iters = 1000;
params.tol = 1e-2;

lambdas = logspace(1,-3,10);
i = str2num(i)+1;
lambda = lambdas(i);

[X Y t s] = problem_synthetic_icml();
St = [Y(t,:) X(t,:)]'*[Y(t,:) X(t,:)]/length(t);
[Lambda Theta] = gcrf_newton(St, lambdas(i), eye(p), zeros(n,p), params);
P  = -X*Theta / Lambda;
err = [mse(P(t,:), Y(t,:)) mse(P(s,:), Y(s,:))];

fprintf('%d\t%f\t%f\n', i, err);

