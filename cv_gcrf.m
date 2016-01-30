[m n] = size(X); p = size(Y,2);

% Test GCRF w/ L1
clear params;
params.quiet = 0;
params.max_iter = 1000;

lambda2 = 1e-4;
clear cv;
cv.lambdas = logspace(1,-3,50);
for i=1:length(cv.lambdas)
  lambda = [cv.lambdas(i) lambda2]
  [Theta Lambda h] = gcrf_admm(X(t,:), Y(t,:), lambda, 1, params); 
  Sigma = inv(Lambda);
  cv.ll(i, :) = [
      log_loss(X(t,:), Y(t,:), Theta, Lambda, Sigma)
      log_loss(X(s,:), Y(s,:), Theta, Lambda, Sigma)];
  cv.mse(i, :) = [
      mse_gcrf(X(t,:), Y(t,:), Theta, Sigma)
      mse_gcrf(X(s,:), Y(s,:), Theta, Sigma)];
  cv.nz(i, :) = [nnz(Theta)/numel(Theta) nnz(Lambda)/numel(Lambda)];
  
  fprintf('%d\t%.4f %.4f\t%.4f %.4f\t%.4f %.4f\n', i, ...
          cv.ll(i,:), cv.mse(i,:), cv.nz(i,:));
end
figure;
semilogx(cv.lambdas, cv.mse);
set(gca, 'xdir', 'reverse');

% Test covariance selection
Dt = [Y(t,:) X(t,:)];
Ds = [Y(s,:) X(s,:)];
clear cv_covsel;
cv_covsel.lambdas = logspace(1,-3,50);
for i=1:1:length(cv_covsel.lambdas)
  Z = covsel(Dt, cv_covsel.lambdas(i), 1, 1);
  Sigma = inv(Z);
  Theta = Sigma(p+1:p+n,p+1:p+n) \ Sigma(p+1:p+n,1:p);
  
  log_det_Z = sum(log(eig(Z)));
  cv_covsel.ll(i,:) = [trace(cov(Dt)*Z) - log_det_Z
                       trace(cov(Ds)*Z) - log_det_Z];
  cv_covsel.mse(i,:) = [mse(X(t,:)*Theta, Y(t,:)) 
                        mse(X(s,:)*Theta, Y(s,:))];
  cv_covsel.nz(i) = [nnz(Z)/numel(Z)];
  
  fprintf('%d\t%f %f\t%f %f\t%f\n', i, ...
          cv_covsel.ll(i,:), cv_covsel.mse(i,:), cv_covsel.nz(i));
end
figure;
semilogx(cv_covsel.lambdas, cv_covsel.mse);
set(gca, 'xdir', 'reverse');


figure;
semilogx(cv_covsel.lambdas, cv_covsel.mse(:,2), 'b', ...
         cv.lambdas, cv.mse(:,2), 'r');
set(gca, 'xdir', 'reverse');

save('sythentic_cv2', 'cv', 'cv_covsel');