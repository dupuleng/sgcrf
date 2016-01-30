function [X Y t s] = problem_loads_icml(path)

load([path 'pjm_loads_2011.txt']);
load([path 'pjm_forecasts_2011.txt']);

[X Y] = features_loads(pjm_loads_2011, pjm_forecasts_2011);
[m n] = size(X); p = size(Y, 2);

% Deal with NaNs by removing rows from labeled data and dropping features.
bad = find(max(isnan(Y), [], 2) | max(isnan(X), [], 2));
X = X(setdiff(1:m, bad), :);
Y = Y(setdiff(1:m, bad), :);
m = m - length(bad);

% Rescale X
X = bsxfun(@rdivide, X, sqrt(diag(X'*X)/m)');

rand('state', 1);
split = cvpartition_grouped(m, 5, 5);
t = find(split.training(:,1));
s = find(split.test(:,1));

% Make training set artificially small
perm = randperm(length(t));
k = 175;
t = t(perm(1:k));


