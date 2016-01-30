function e = rmse(Y, P)
  e = sqrt(mean(mean((Y-P).^2)));
end