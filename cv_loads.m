
[X Y t s] = problem_loads_icml('../data/');

clear cv;
cv.lambdas_l2 = logspace(5,-5,50);
cv.lambdas_l1 = logspace(1,-2,10);
cv.params.max_iters = 100;
cv.params.tol = 1e-2;
cv.params.stop_early = 1;

cv.ls = cv_ls(X(t,:), Y(t,:), X(s,:), Y(s,:), cv.lambdas_l2);
%cv.crf = cv_crf(X(t,:), Y(t,:), X(s,:), Y(s,:), cv.lambdas_l1, cv.params);

%save('cv_loads', 'cv');