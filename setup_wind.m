% Wind data set experiments for ICML

wind = csvread('../data/wind/train.csv', 1, 0);
for i=1:7,
  wf{i} = csvread(['../data/wind/windforecasts_wf' num2str(i) '.csv'], 1, 0);
  wf{i}(wf{i} == -10000) = nan;
  wf{i}(:,1) = date_to_timestamp(wf{i}(:,1));
end

[X Y] = problem_wind(wind, wf);
clear params;
params.num_ar_features = 7*8;
params.rbf_mus = 10;
params.bandwidth = 7;
Phi = features_wind(X, params, struct);
[m p] = size(Y); n = size(Phi, 2);

% Normalize columns and give Y's mean zero
Phi = bsxfun(@rdivide, Phi, sqrt(diag(Phi'*Phi)/m)');
Y = bsxfun(@minus, Y, mean(Y));

rand('state', 1);
split = cvpartition_grouped(m, 5, 8);
t = find(split.training(:,1));
s = find(split.test(:,1));

