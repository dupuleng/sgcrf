
clear params;
[X Y] = problem_wind_small(wind, wf);
params.num_ar_features = 7*8;
params.rbf_mus = 10;
params.bandwidth = 7;
Phi = features_wind(X, params, struct);
[m p] = size(Y); n = size(Phi,2);

rand('state', 1);
split = cvpartition_grouped(m, 5, 4);
t = find(split.training(:,1));
s = find(split.test(:,1));

% Fit a linear model
lambda2 = 1e-2;
E = length(t)*lambda2*diag([ones(n-1,1); 0]);
Theta = (Phi(t,:)'*Phi(t,:) + E) \ (Phi(t,:)'*Y(t,:));
P = Phi*Theta;
Sigma = diag(var(Y(t,:)-P(t,:)));

% Fit a linear model w/ copulas and 95% confidence interval
Y_norm = copula(Y(t,:));
Theta2 = (Phi(t,:)'*Phi(t,:) + E) \ (Phi(t,:)'*Y_norm);
P2_norm = Phi*Theta2;
Sigma2_norm = diag(var(Y(t,:)-P2_norm(t,:)));

% Choose the interval based to get 95% coverage
P2 = inv_copula(Y(t,:), P2_norm);
P2_hi = inv_copula(Y(t,:), P2_norm + 2);
P2_lo = inv_copula(Y(t,:), P2_norm - 2);

% Fit gcrf model
clear params;
params.max_iter = 100;
params.quiet = 0;
[Theta3 Lambda3 h] = gcrf_admm(Phi(t,:), Y(t,:), [3e-3 1e-4], 1, params);

% Fit gcrf model - LS
[Theta3 Lambda3] = gcrf_ls(Phi(t,:), Y_norm, [0 1e-4]);

Sigma3_norm = inv(Lambda3); 
Sigma3_norm = (Sigma3_norm + Sigma3_norm')/2;
P3_norm = -Phi*Theta3*Sigma3_norm;
P3 = inv_copula(Y(t,:), P3_norm);
mse(P3(s,:), Y(s,:))

% Generate a histogram by sampling from each distribution
idx = 2;

for i=1:1
  Yp = mvnrnd(repmat(P(s(i),:),10000,1), Sigma);
  Yp2_norm = mvnrnd(repmat(P2_norm(s(i),:),10000,1),Sigma2_norm);
  Yp2 = inv_copula(Y(t,:), Yp2_norm);
  Yp3_norm = mvnrnd(repmat(P3_norm(s(i),:),10000,1),Sigma3_norm);
  Yp3 = inv_copula(Y(t,:), Yp3_norm);
  
  [n x] = hist(sum(Yp,2), 100);
  [n2 x2] = hist(sum(Yp2,2), 100);  
  [n3 x3] = hist(sum(Yp3,2), 100);
  
  figure;
  plot(x, n, 'b', x2, n2, 'r', x3, n3, 'g');
end


figure;
plot(1:24, P(s(1),1:24), 'b', ...
     1:24, P2(s(1),1:24), 'r');


             


