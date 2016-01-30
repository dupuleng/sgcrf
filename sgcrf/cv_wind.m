[X Y t s] = problem_wind_icml('../data/');

% Comparison of CRF, MRF and LS 
clear cv;
cv.lambdas_l2 = logspace(5,-5,50);
cv.lambdas_l1 = logspace(1,-3,20);

clear params;
params.max_iters = 1000;
params.epsilon = 1e-2;

cv.ls = cv_ls(X(t,:), Y(t,:), X(s,:), Y(s,:), cv.lambdas_l2);
cv.crf = cv_crf(X(t,:), Y(t,:), X(s,:), Y(s,:), cv.lambdas_l1, params);

save('cv_wind', 'cv');