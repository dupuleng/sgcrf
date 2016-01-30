function [Lambda Theta] = gen_params(n, p)

Theta = sprandn(n, p, 10/n);
Lambda = eye(p);
idx = randsample(p^2, 5*p);
Lambda(idx) = sign(rand(numel(idx),1)*2-1);
Lambda = Lambda' + Lambda;
% Give Lambda condition number p/10
c = floor(p/10);
v = eig(Lambda);
sigma = (max(v) - c*min(v)) / (c-1);
Lambda = Lambda + sigma*eye(p);

