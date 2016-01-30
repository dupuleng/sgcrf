
[X Y t s] = problem_synthetic_icml();

clear cv;
lambdas_l2 = logspace(5,-5,50);
lambdas_l1 = logspace(1,-3,20);

clear params;
params.max_iters = 100;
params.tol = 1e-2;

cv_ls = cv_ls(X(t,:), Y(t,:), X(s,:), Y(s,:), cv.lambdas_l2);
cv_crf = cv_crf(X(t,:), Y(t,:), X(s,:), Y(s,:), cv.lambdas_l1, cv.params);
cv_mrf = cv_mrf(X(t,:), Y(t,:), X(s,:), Y(s,:), cv.lambdas_l1, cv.params.epsilon, ...
                cv.params.max_iters*length(cv.lambdas_l1));


