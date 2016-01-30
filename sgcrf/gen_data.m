function [X Y] = gen_data(Lambda, Theta, m)
n = size(Theta, 1);
A = randn(n);
X = randn(m,n)*A/sqrt(n);
X(:, 1:n/2) = X(:, 1:n/2).^2;
X = bsxfun(@rdivide, X, sqrt(diag(X'*X)/m)');
Sigma = inv(Lambda);
Y = mvnrnd(-X*Theta*Sigma, Sigma);






