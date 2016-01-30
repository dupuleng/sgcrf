[X Y] = problem_wind_small_ar(wind, wf);
clear params;
params.num_ar_features = 7*8;
params.rbf_mus = 10;
params.bandwidth = 7;
Phi = features_vector(X, params, struct);
[m p] = size(Y); n = size(Phi,2);

% Leave out the last 50 days for testing
t = 1:11928;
s = 11929:13128;

p = 1;
clear model;
for k=1:7
  data = iddata(Y(:,k),Phi);
  model{k} = armax(data(t), [3*ones(p,p), ones(p,n) 2*ones(p,1), zeros(p,n)]);
end

mse = zeros(50,7);
for j=1:50
  for k=1:7
    i = (j-1)*24+1;
    P = ar_predict(model{k}, Phi(s(i):s(i)+23,:), Y(s(i)-24:s(i)-1,k));
    mse(j,k) = mean((Y(s(i):s(i)+23,k) - P).^2);
  end
  fprintf('%d\t%f\n', j, sqrt(mean(mse(j,:))));
end
  
coverage = zeros(50,3);
coverage_l = zeros(50*24,3);
coverage_t = zeros(50*7,3);

Yp = zeros(10000, 24*7);
Yi = zeros(24*7, 1);
alphas = [.1 .05 .01];
for j=1:50
  i = (j-1)*24+1;
  for k=1:7
    Yp(:,(k-1)*24+1:k*24) = ar_sample(model{k}, 10000, Phi(s(i):s(i)+23,:), ...
                                      Y(s(i)-24:s(i)-1, k));
    Yi((k-1)*24+1:k*24) = Y(s(i):s(i)+23,k);
  end
  
  coverage(j,:) = check_coverage(sum(Yp,2), sum(Yi), alphas);
  
  for l=1:7
    idx = (l-1)*24+1:l*24;
    coverage_t((j-1)*7+l,:) = ...
        check_coverage(sum(Yp(:,idx),2), sum(Yi(idx)), alphas);
  end
  
  for l=1:24
    idx = l:24:7*24;
    coverage_l((j-1)*24+l,:) = ...
        check_coverage(sum(Yp(:,idx),2), sum(Yi(idx)), alphas);
  end
  
  fprintf('%d\t%f/%f/%f\n', j, sum(coverage,1)/j);
end

fprintf('%f/%f/%f\t%f/%f/%f\t%f/%f/%f\n', mean(coverage), mean(coverage_l), ...
        mean(coverage_t));

