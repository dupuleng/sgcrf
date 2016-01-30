[X Y] = problem_wind_icml('../data/');

% Comparison of CRF, MRF and LS 
lambda_l1 = 0.05;
lambda_l2 = 3.2;

params.max_iters = 1000;
params.epsilon = 1e-2;

rand('state', 1);
split = cvpartition_grouped(size(X,1), 5, 8);

clear kfold;
for i=1:5
  t = find(split.training(:,i));
  s = find(split.test(:,i));

  ls = cv_ls(X(t,:), Y(t,:), X(s,:), Y(s,:), lambda_l2);
  kfold.ls(i,:) = ls.mse(1,:);
  kfold.crf(i,:) = cv_crf_warm(X(t,:), Y(t,:), X(s,:), Y(s,:), lambda_l1, params);
end

save('cv_wind_kfold', 'lambda_l1', 'lambda_l2', 'params', 'kfold');