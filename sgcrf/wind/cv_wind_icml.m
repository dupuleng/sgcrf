
[X Y] = problem_wind_small(wind, wf);
clear params;
params.num_ar_features = 7*8;
params.rbf_mus = 10;
params.bandwidth = 7;
Phi = features_wind(X, params, struct);
[m p] = size(Y); n = size(Phi,2);

rand('state', 1);
split = cvpartition_grouped(m, 5, 4);
t = find(split.training(:,1));
s = find(split.test(:,1));
S = [Y(t,:) Phi(t,:)]'*[Y(t,:) Phi(t,:)]/length(t);

% CV for linear model
clear cv;
cv.linear.lambdas = logspace(5,-5,50);
for i=1:length(cv.linear.lambdas)
  [Lambda Theta] = gcrf_ls(S + cv.linear.lambdas(i)*eye(n+p), p);
  P = -Phi*Theta / Lambda;
  cv.linear.rmse(i,:) = [rmse(P(t,:), Y(t,:)) rmse(P(s,:), Y(s,:))];  
  
  fprintf('%d\t%f\t%f\n', i, cv.linear.rmse(i,:));
end

figure;
semilogx(cv.linear.lambdas, cv.linear.rmse, 'LineWidth', 1);
legend('Train', 'Test');
set(gca, 'xdir', 'reverse');
prepare_figure('cv_wind_linear.pdf', [4 3], '\lambda', 'RMSE')

% CV for GCRF
clear params;
params.tol = 1e-2;
params.max_iters = 1000;
cv.gcrf.lambdas = logspace(0,-4,20);
Lambda = eye(p);
Theta = zeros(n,p);
for i=1:length(cv.gcrf.lambdas)
  [Lambda Theta] = gcrf_newton(S, cv.gcrf.lambdas(i), Lambda, Theta, params);
  P  = -Phi*Theta / Lambda;
  cv.gcrf.rmse(i,:) = [rmse(P(t,:), Y(t,:)) rmse(P(s,:), Y(s,:))];
  fprintf('%d\t%f\t%f\n', i, cv.gcrf.rmse(i,:));
end

figure;
semilogx(cv.gcrf.lambdas, cv.gcrf.rmse, 'LineWidth', 1);
legend('Train', 'Test');
set(gca, 'xdir', 'reverse');
prepare_figure('cv_wind_gcrf.pdf', [4 3], '\lambda', 'RMSE')

baseline = min(cv.linear.rmse(:,2));
figure;
semilogx(cv.gcrf.lambdas, cv.gcrf.rmse(:,2), ...
         cv.gcrf.lambdas, ones(length(cv.gcrf.lambdas),1)*baseline, 'k--');
legend('GCRF', 'Baseline');
set(gca, 'xdir', 'reverse');
xlim([1e-4 1e-1])
prepare_figure('cv_wind_comparison.pdf', [4 3], '\lambda', 'RMSE');

