
% Perform cross-validation for the LS and SGCRF models used in the PES/CDC
% papers on wind forecasting. 
[X Y t s] = problem_wind_cdc('../data/');

% Sparse GCRF Newton algorithm parameters
clear params;
params.max_iters = 300;
params.tol = 1e-2;

%clear cv;
%cv.ls = cv_ls(X(t,:), Y(t,:), X(s,:), Y(s,:), logspace(2,-5, 10));
cv.crf = cv_crf(X(t,:), Y(t,:), X(s,:), Y(s,:), logspace(-1, -3, 10), params);

% Copula transform
cv.ls_copula = cv_ls_copula(X(t,:), Y(t,:), X(s,:), Y(s,:), logspace(2,-2, 2));
cv.crf_copula = cv_crf_copula(X(t,:), Y(t,:), X(s,:), Y(s,:), logspace(-1,-3,2), ...
                              params);

% Save the CV curves and best models 
save('cv_wind_models', 'cv');
