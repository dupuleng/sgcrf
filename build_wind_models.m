
% Build the models using the PES/CDC papers on wind forecasting.
[X Y t s] = problem_wind_icml('../data/');

clear params;
params.max_iters = 100;
params.tol = 1e-2;
params.stop_early = 1;

clear cv;
cv.ls = cv_ls(X(t,:), Y(t,:), X(s,:), Y(s,:), logspace(5,-5,10));
cv.crf = cv_crf(X(t,:), Y(t,:), X(s,:), Y(s,:), logspace(-1, -3, 10), params);

% Copula transform
Y_norm = copula(Y(t,:));



lambda2 = 2e-2;
[Theta Lambda] = gcrf_ls(Phi(t,:), Y(t,:), [0 2e-2]);
P = -Phi*Theta / Lambda;
mse1 = [rmse(P(t,:), Y(t,:)) rmse(P(s,:), Y(s,:))]

% LR w/ copulas - choose lambda by cross validation
lambda = 2e-2;
Y_norm = copula(Y(t,:));
[Theta2 Lambda2] = gcrf_ls(Phi(t,:), Y_norm, [0 2e-2]);
P2_norm = -Phi*Theta2 / Lambda2;
P2 = inv_copula(Y(t,:), P2_norm);
mse2 = [rmse(P2(t,:), Y(t,:)) rmse(P2(s,:), Y(s,:))]

% Sparse GCRF 


% Sparse GCRF w/ copulas
clear params;
params.max_iter = 1000;
params.quiet = 0;
[Theta3 Lambda3 h] = gcrf_admm(Phi(t,:), Y_norm, [3e-3 1e-4], 1, params);
Sigma3_norm = inv(Lambda3); 
Sigma3_norm = (Sigma3_norm + Sigma3_norm')/2;
P3_norm = -Phi*Theta3*Sigma3_norm;
P3 = inv_copula(Y(t,:), P3_norm);
mse3 = [rmse(P3(t,:), Y(t,:)) rmse(P3(s,:), Y(s,:))]
             
save('model', 't', 's', ...
     'Theta', 'Lambda', ...
     'Theta2', 'Lambda2', ...
     'Theta3', 'Lambda3', 'h');
