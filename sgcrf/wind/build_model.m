
clear params;
[X Y] = problem_wind_icml(wind, wf);
params.num_ar_features = 7*8;
params.rbf_mus = 10;
params.bandwidth = 7;
Phi = features_wind(X, params, struct);
[m p] = size(Y); n = size(Phi,2);

rand('state', 1);
split = cvpartition_grouped(m, 5, 4);
t = find(split.training(:,1));
s = find(split.test(:,1));

% GCRF
lambda2 = 2e-2;
[Theta Lambda] = gcrf_ls(Phi(t,:), Y(t,:), [0 2e-2]);
P = -Phi*Theta / Lambda;
mse1 = [rmse(P(t,:), Y(t,:)) rmse(P(s,:), Y(s,:))]

% GCRF w/ copulas 
lambda = 2e-2;
Y_norm = copula(Y(t,:));
[Theta2 Lambda2] = gcrf_ls(Phi(t,:), Y_norm, [0 2e-2]);
P2_norm = -Phi*Theta2 / Lambda2;
P2 = inv_copula(Y(t,:), P2_norm);
mse2 = [rmse(P2(t,:), Y(t,:)) rmse(P2(s,:), Y(s,:))]

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
