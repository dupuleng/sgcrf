
[X Y] = problem_wind_small(wind, wf);

% Linear model
P = -Phi*Theta / Lambda;
Sigma = cov(Y(t,:) - P(t,:));
mse1 = [rmse(P(t,:), Y(t,:)) rmse(P(s,1:24), Y(s,1:24))]

% Linear model w/ copulas 
Y_norm = copula(Y(t,:));
P2_norm = -Phi*Theta2 / Lambda2;
Sigma2 = cov(Y_norm - P2_norm(t,:));
P2 = inv_copula(Y(t,:), P2_norm);
mse2 = [rmse(P2(t,:), Y(t,:)) rmse(P2(s,1:24), Y(s,1:24))]

% gcrf model
Sigma3 = inv(Lambda3); 
Sigma3 = (Sigma3 + Sigma3')/2;
P3_norm = -Phi*Theta3*Sigma3;
P3 = inv_copula(Y(t,:), P3_norm);
mse3 = [rmse(P3(t,:), Y(t,:)) rmse(P3(s,1:24), Y(s,1:24))]

% Generate a histogram by sampling from each distribution and see how often
% it captures the true point
coverage = zeros(length(s),9);
coverage_l = zeros(length(s)*24,9);
coverage_t = zeros(length(s)*7,9);
alphas = [.1 .05 .01];
for i=1:length(s)
  Yp = mvnrnd(repmat(P(s(i),:),10000,1), Sigma);
  Yp2_norm = mvnrnd(repmat(P2_norm(s(i),:),10000,1),Sigma2);
  Yp2 = inv_copula(Y(t,:), Yp2_norm);
  Yp3_norm = mvnrnd(repmat(P3_norm(s(i),:),10000,1),Sigma3);
  Yp3 = inv_copula(Y(t,:), Yp3_norm);
  
  coverage(i,:) = [
      check_coverage(sum(Yp,2), sum(Y(s(i),:),2), alphas)
      check_coverage(sum(Yp2,2), sum(Y(s(i),:),2), alphas)
      check_coverage(sum(Yp3,2), sum(Y(s(i),:),2), alphas)];

  % For every location, sum across all hours
  for j=1:7
    idx = (j-1)*24+1:j*24;
    coverage_t((i-1)*7+j,:) = [
      check_coverage(sum(Yp(:,idx),2), sum(Y(s(i),idx),2), alphas)
      check_coverage(sum(Yp2(:,idx),2), sum(Y(s(i),idx),2), alphas)
      check_coverage(sum(Yp3(:,idx),2), sum(Y(s(i),idx),2), alphas)];      
  end

  % For every hour, sum across all locations
  for j=1:24
    idx = j:24:7*24;
    coverage_l((i-1)*24+j,:) = [
      check_coverage(sum(Yp(:,idx),2), sum(Y(s(i),idx),2), alphas)
      check_coverage(sum(Yp2(:,idx),2), sum(Y(s(i),idx),2), alphas)
      check_coverage(sum(Yp3(:,idx),2), sum(Y(s(i),idx),2), alphas)];
  end

  fprintf('%d\t%f/%f/%f\t%f/%f/%f\t%f/%f/%f\n', i, sum(coverage,1)/i);
end




