function [X Y w] = problem_forecast(loads, temps, zone)

% Create the hourly problem
[Xh Yh] = problem_hourly(loads, temps);
Yh = Yh(:,zone);

% one training example for each day
m = floor(Xh.m/24);

% Pad up input data with a weeks worth of NaNs on both sides
Yh_pad = [nan(168,1); Yh; nan(168,1)];
X_temp_pad = [nan(168,11); Xh.temp; nan(168,11)];

Y = nan(m,168);
X.prev_week = nan(m,168);
X.prev_temp = nan(m,11*168);
X.temp = nan(m,11*168);

for i=1:m
  % Calculate the offset into the hourly vector
  j = (i-1)*24+169;
  k = j+167;
  
  % output data, week long time series
  Y(i,:) = Yh_pad(j:k);
  
  % Data for the previous week, see note above
  X.prev_week(i,:) = Yh_pad(j-168:k-168);
  X.prev_temp(i,:) = vec(X_temp_pad(j-168:k-168, :));
  X.temp(i,:) = vec(X_temp_pad(j:k, :));
end

j = ([1:m]-1)*24+1;
X.year = Xh.year(j);
X.time = Xh.time(j);
X.week = Xh.week(j);
X.m = m;

% Scale the input features to make them N(0,1)
mu = nanmean(Y(:));
sigma = nanstd(Y(:));
X.prev_week = (X.prev_week-mu)/sigma;

% weight things based on time of the year, the week we care about starts about
% hour 4400. note that this hour is in the middle of the year, so ignore the
% fact that the year is cyclical.
target_hour = 4400;
d = (X.year-target_hour).^2;
h = 24*7*8;
%w = exp(-0.5/h^2 * d);
w = ones(m,1);
