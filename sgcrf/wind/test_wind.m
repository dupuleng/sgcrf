

clear params;
params.num_ar_features = 7*8;
params.rbf_mus = 10;
params.bandwidth = 7;
[X Y] = problem_wind_small(wind, wf);
Phi = features_wind(X, params, struct);

[m p] = size(Y);
rand('state', 1);
cv = cvpartition_grouped(m, 5, 4);
t = find(cv.training(:,1));
s = find(cv.test(:,1));

lambda = [1e-4 1e-2];

clear params;
params.max_iter = 100;
params.quiet = 0;
[Theta0 Lambda0 h0] = gcrf_admm(Phi, Y, lambda, 1, params);
loss0 = [log_loss(Phi(t,:), Y(t,:), Theta0, Lambda0, inv(Lambda0)), ...
         log_loss(Phi(s,:), Y(s,:), Theta0, Lambdd0, inv(Lambda0))]

clear params;
params.max_iter = 10;
params.quiet = 0;
[Theta1 Lambda1 h1] = gcrf_admm2(Phi(t,:), Y(t,:), lambda, 0.1, params);
loss1 = [log_loss(Phi(t,:), Y(t,:), Theta1, Lambda1, inv(Lambda1)), ...
         log_loss(Phi(s,:), Y(s,:), Theta1, Lambda1, inv(Lambda1))]
err1 = [mse(Phi(t,:), Y(t,:), Theta1, inv(Lambda1)) ...
        mse(Phi(s,:), Y(s,:), Theta1, inv(Lambda1))]
            
         

