function [X Y] = problem_hourly(loads, temps)

loads_days = 1650;
loads_locations = 20;

temps_days = 1643;
temps_locations = 11;

% reshape temps to have row per hour and column per location
for i=1:temps_locations
  j = (i-1)*temps_days+1;
  k = i*temps_days;
  X.temp(:, i) = reshape(temps(j:k, 5:end)', [], 1);
end

% for each location create a separate data set
for i=1:loads_locations
  j = (i-1)*loads_days+1;
  k = i*loads_days;
  Y(:, i) = reshape(loads(j:k, 5:end)', [], 1);
end

m = loads_days*24;

% pad missing temperature data and rescale it
X.temp(end+1:m, :) = NaN;
X.temp_u = X.temp;
X.temp = bsxfun(@rdivide, bsxfun(@minus, X.temp, nanmean(X.temp)), ...
                nanstd(X.temp));

% periodic time features that repeat every day, every week and every year.
X.day = repmat([1:24]', loads_days, 1);
X.week = repmat([1:24*7]', ceil(loads_days/7), 1); X.week = X.week(1:m);
X.year = repmat([1:24*365]', ceil(loads_days/365), 1); X.year = X.year(1:m);

% monotonic time feature
X.time = [1:1650*24]' / (1650*24);

% holidays
hdays = holidays();
X.holidays = hdays(ceil([1:m]/24));

X.m = m;
X.p = loads_locations;

Yall = sum(Y,2);
Y = [Y Yall];