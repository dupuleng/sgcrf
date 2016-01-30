function [X Y] = problem_wind_small(wind, wf)
% Create a wind power forecasting problem with input variables X and output
% variables Y. There is one example per 48-hour wind forecast and the input
% features include the forecast as well as 8 hour trailing information for
% wind power production. 

t0 = wf{1}(:,1);
[t h] = date_to_timestamp(wind(:,1));

idx = find(wf{1}(:,2) == 1 & t0 < datenum(2010,12,30));
idx = idx(2:end);
m = length(idx);
idx2 = zeros(m,1);
for i=1:m
  idx2(i) = find(t == t0(idx(i)));
end

X = zeros(m, 7*8+7*24*3);
Y = zeros(m, 7*24);

for i=1:m 
  Y(i,:) = vec(wind(idx2(i)+1:idx2(i)+24, 2:8));
  X(i,1:7*8) = vec(wind(idx2(i)-7:idx2(i), 2:8));
  
  for j=1:length(wf)
    X(i,7*8+(j-1)*24*3+1:7*8+j*24*3) = vec(wf{j}(idx(i):idx(i)+23,3:5)');
  end
end
