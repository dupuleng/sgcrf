
% loads measured every 6 hours
loads6 = loads(6:6:end, :);



% find all data points with no NaNs
good = setdiff(1:length(forecasts), find(isnan(max(forecasts, [], 2))));

% MSE by window of prediction
for i=1:25
  mse(i, :) = mean((loads6(good, :) - forecasts(good, i:25:end)).^2);
end

legend

plot(mse)
