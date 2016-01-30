[X Y] = features(loads, forecasts);
[m n] = size(X); p = size(Y, 2);

% Deal with NaNs by removing rows from labeled data and dropping features.
bad = find(max(isnan(Y), [], 2));
X(isnan(X)) = 0;

t = setdiff(1:28, bad);
s = 29:56;

% lambda = [0 1e-5];
% [Theta_ls Lambda_ls] = gcrf_ls(X, Y, lambda);
% loss_ls = objective(X, Y, Theta_ls, Lambda_ls, inv(Lambda_ls), lambda)

% [Theta Lambda] = gcrf_admm(X, Y, lambda);
% loss_admm = objective(X, Y, Theta, Lambda, inv(Lambda), lambda)

lambda2 = 1e-5;
lambdas=logspace(0, -5, 50);
clear cv;

clear params;
params.max_iter = 1000;

for i=1:length(lambdas)
  lambda = [lambdas(i) lambda2]
  [Theta Lambda h] = gcrf_admm(X(t,:), Y(t,:), lambda, 0.2, params);
    
  Sigma = inv(Lambda);
  cv.ll(i, :) = [
      log_loss(X(t,:), Y(t,:), Theta, Lambda, Sigma)
      log_loss(X(s,:), Y(s,:), Theta, Lambda, Sigma)];
  cv.mse(i, :) = [
      mse(X(t,:), Y(t,:), Theta, Sigma)
      mse(X(s,:), Y(s,:), Theta, Sigma)];
  
  figure;
  k = length(h.r_norm);
  ax = semilogy(1:k, h.r_norm, 'r', ...
                1:k, h.s_norm, 'b');
  legend([ax(1) ax(4)], 'r_k', 's_k');
  
  cv.ll
end


