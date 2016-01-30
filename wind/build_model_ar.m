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

data = iddata(Y,Phi);
%model = arx(data(t),[3*ones(p,p) 2*ones(p,n) zeros(p,n)]);
model = armax(data(t), [ones(p,p), ones(p,n) ones(p,1), zeros(p,n)]);

mse = zeros(24,1);
for k=1:24
  yp = predict(model, data, k);
  P = yp.y;
  mse(k) = mean(mean((P(s,:) - Y(s,:)).^2));
  fprintf('%d\t%f\n', k, sqrt(mse(k)));
end
sqrt(mean(mse))

figure;
plot(1:24, sqrt(mse1), 'b', ...
     1:24, sqrt(mse), 'r');