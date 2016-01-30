function [Phi Y] = features_loads(loads, forecasts) 
  [m_loads num_locations] = size(loads);
  [m_forecasts forecast_locations] = size(forecasts);

  % number of forecats per location
  num_forecasts = 25;
  num_historical = num_forecasts;

  % Scale data to have zero mean and unit variance
  mu = nanmean(loads);
  sigma = nanstd(loads);
  loads = bsxfun(@rdivide, bsxfun(@minus, loads, mu), sigma);

  m = m_forecasts;
  Y = zeros(m, num_locations*num_forecasts);
  X_hour = zeros(m, 3);
  X_day = zeros(m, 6);
  X_past_day = zeros(m, num_locations*num_historical);
  X_past_week = zeros(m, num_locations*num_historical);
  for i=1:m
    % forecasts are every 6 hours, loads are every hour. 
    j = i*6;  
    
    % create loads response data, pad with NaNs for missing.
    k = min(j+num_forecasts, m_loads+1);
    num_missing = num_forecasts - (k - j);
    A = [loads(j:k-1, :); ones(num_missing, num_locations)*NaN];
    Y(i, :) = reshape(A, 1, num_locations*num_forecasts);
               
    % add categorical features for hour of day 
    h = zeros(1, 4);
    h(mod(i-1, 4)+1) = 1;
    X_hour(i, :) = h(1:3);
    
    d = zeros(1, 7);
    d(mod(floor((i-1)/4), 7)+1) = 1;
    X_day(i, :) = d(1:6);
        
    % add historical data for trailing day
    k = max(1, j-num_historical);
    num_missing = num_historical - (j - k);
    A = [ones(num_missing, num_locations)*NaN; loads(k:j-1, :)];
    X_past_day(i, :) = reshape(A, 1, num_locations*num_historical);
    
    % add historical data for same day last week
    j = max(1, j - 24*7 + 1);
    k = max(1, j - num_historical);
    num_missing = num_historical - (j - k);
    A = [ones(num_missing, num_locations)*NaN; loads(k:j-1, :)];
    X_past_week(i, :) = reshape(A, 1, num_locations*num_historical);
  end

  % concatenate all the feature vectors and add bias term
  Phi = [forecasts X_past_day X_past_week X_hour X_day ones(m, 1)];
end