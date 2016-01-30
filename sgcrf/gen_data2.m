function [X Y] = gen_data2(Omega, m, p)
% Generate data using the procedure described in Yuan and Zhang
n = size(Omega,1) - p;

% Generate X, Y jointly
% Z = mvnrnd(zeros(m, n+p), inv(Omega));
% Y = Z(:, 1:p);
% X = Z(:, p+1:p+n);

% Generate X first
%Sigma = inv(Omega);
%X = mvnrnd(zeros(m, n), Sigma(p+1:p+n, p+1:p+n));

A = randn(n);
X = randn(m,n)*A/sqrt(n);
Theta = Omega(p+1:p+n, 1:p);
Lambda = Omega(1:p, 1:p);
Y = mvnrnd(-X*Theta*inv(Lambda), inv(Lambda));

