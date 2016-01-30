% Synthetic data set experimental for ICML

addpath ../pred;
addpath newton;
addpath fista;
addpath owlqn;

m = 2500;
n = 4000;
p = 1000;
lambda = 0.2;

randn('seed', 1);
rand('seed', 1);

[Lambda0 Theta0] = gen_params3(n, p);
[X Y] = gen_data(Lambda0, Theta0, m*2);

% Normalize columns and give Y's mean zero
X = bsxfun(@rdivide, X, sqrt(diag(X'*X)/m)');
Y = bsxfun(@minus, Y, mean(Y));

t = 1:m;
s = m+1:2*m;

