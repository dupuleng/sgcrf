
m = 100;
n = 10;
p = 5;
lambda = 0.1;

clear params;
params.d_tol = 1e-5;

rand('seed', 1);
randn('seed', 1);
S = cov([randn(m,p) rand(m,n)*2-1]);

disp('Initial');
Lambda = eye(p);
Theta = zeros(n,p);
Sigma = inv(Lambda);
[U V] = argmin_dir(S, lambda, Lambda, Theta, Sigma, params);
test_dir_minimal(Lambda, Theta, U, V, S, lambda, Sigma);

disp('Random Theta');
Theta = rand(n,p)*2-1;
Lambda = eye(p);
Sigma = inv(Lambda);
[U V] = argmin_dir(S, lambda, Lambda, Theta, Sigma, params);
test_dir_minimal(Lambda, Theta, U, V, S, lambda, Sigma);

disp('Random Lambda');
Theta = zeros(n,p);
Lambda = rand(p,p)*2-1;
Lambda = Lambda' + Lambda;
Lambda = Lambda + 1.1*abs(min(eig(Lambda)))*eye(p);
Sigma = inv(Lambda);
[U V] = argmin_dir(S, lambda, Lambda, Theta, Sigma, params);
test_dir_minimal(Lambda, Theta, U, V, S, lambda, Sigma);

disp('Random Theta and Lambda');
Theta = rand(n,p)*2-1;
Lambda = rand(p,p)*2-1;
Lambda = Lambda' + Lambda;
Lambda = Lambda + 1.1*abs(min(eig(Lambda)))*eye(p);
Sigma = inv(Lambda);
[U V] = argmin_dir(S, lambda, Lambda, Theta, Sigma, params);
test_dir_minimal(Lambda, Theta, U, V, S, lambda, Sigma);

disp('PASSED');
