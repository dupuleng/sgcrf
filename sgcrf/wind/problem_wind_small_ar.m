function [X Y] = problem_wind_small_ar(wind, wf)
% Create a wind power forecasting problem with input variables X and output
% variables Y. There is one example per 48-hour wind forecast and the input
% features include the forecast as well as 8 hour trailing information for
% wind power production. 

t0 = wf{1}(:,1);
[t h] = date_to_timestamp(wind(:,1));

idx = find(wf{1}(:,2) == 1 & t0 < datenum(2010,12,30));
idx = idx(2:2:end);
idx2 = zeros(length(idx),1);
for i=1:length(idx)
  idx2(i) = find(t == t0(idx(i)));
end

Y = wind(idx2(1)+1:idx2(end)+24, 2:8);
X = zeros(length(idx)*24,7*3);
for i=1:length(idx)
  for j=1:length(wf)
    X((i-1)*24+1:i*24,3*(j-1)+1:j*3) = wf{j}(idx(i):idx(i)+23,3:5);
  end
end
