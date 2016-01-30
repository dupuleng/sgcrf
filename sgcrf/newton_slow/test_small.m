randn('seed', 0);
rand('seed', 0);

m = 100;
n = 10;
p = 5;
lambda = 0.1;

rand('seed', 1);
randn('seed', 1);
Y = randn(m,p);
X = rand(m,n)*2-1;
S = cov([Y X]);

% Solve with CVX
cvx_begin
    variable Z(p+n,p+n) symmetric;
    minimize( -log_det(Z(1:p, 1:p)) + trace(Z*S) + ...
              lambda*sum(sum(abs(Z(1:p, 1:p) - diag(diag(Z(1:p, 1:p)))))) ...
              + lambda*sum(sum(abs(Z(p+1:p+n, 1:p)))) );
    subject to
      Z == semidefinite(p+n);
cvx_end

% Sanity check the parameters
Lambda0 = Z(1:p, 1:p);
Theta0 = Z(p+1:p+n, 1:p);
objval = objective(S, lambda, Lambda0, Theta0, chol(Lambda0), inv(Lambda0));
assert(abs(cvx_optval - objval) < 1e-4);

% Solve with Newton method
clear params;
params.epsilon = 1e-4;
params.max_iters = 10;
[Lambda Theta h] = gcrf_newton(X, Y, lambda, params);
assert(abs(cvx_optval - h.objval(end)) < params.epsilon);

disp('PASSED');