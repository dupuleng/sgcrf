function dist_size_synthetic(idx)

ks = linspace(0.1, 1, 10);
lambdas_l1 = logspace(1,-3,10);
lambdas_l2 = logspace(5,-5,50);
params.max_iters = 1000;
params.epsilon = 1e-2;

%% 

[X Y t s] = problem_synthetic_icml();

% Make training set artificially small
perm = randperm(length(t));
k = floor(length(t) * ks(str2num(idx)+1));
t = t(perm(1:k));
fprintf('Size m=%f\n', k);

cv = cv_ls(X(t,:), Y(t,:), X(s,:), Y(s,:), lambdas_l2);
fprintf('Best L2: %f\n', min(cv.mse(:,2)));

cv = cv_crf(X(t,:), Y(t,:), X(s,:), Y(s,:), lambdas_l1, params);
fprintf('Best L1: %f\n', min(cv.mse(:,2)));

