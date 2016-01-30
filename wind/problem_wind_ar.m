function [X Y] = problem_wind_ar(path)

wind = csvread([path 'wind/train.csv'], 1, 0);
for i=1:7,
  wf{i} = csvread([path 'wind/windforecasts_wf' num2str(i) '.csv'], 1, 0);
  wf{i}(wf{i} == -10000) = nan;
  wf{i}(:,1) = date_to_timestamp(wf{i}(:,1));
end

t0 = wf{1}(:,1);
[t h] = date_to_timestamp(wind(:,1));

idx = find(wf{1}(:,2) == 1 & t0 < datenum(2010,12,30));
idx = idx(2:4:end);
idx2 = zeros(length(idx),1);
for i=1:length(idx)
  idx2(i) = find(t == t0(idx(i)));
end

Y = wind(idx2(1)+1:idx2(end)+48, 2:8);
X = zeros(length(idx)*48,7*3);
for i=1:length(idx)
  for j=1:length(wf)
    X((i-1)*48+1:i*48,3*(j-1)+1:j*3) = wf{j}(idx(i):idx(i)+47,3:5);
  end
end

clear params;
params.rbf_mus = 10;
params.bandwidth = 7;
X = features_vector(X, params, struct);

% Normalize columns
X(:,1:end-1) = bsxfun(@rdivide, bsxfun(@minus, X(:,1:end-1),  mean(X(:,1:end-1))), ...
                      std(X(:,1:end-1)));
Y = bsxfun(@minus, Y, mean(Y));


