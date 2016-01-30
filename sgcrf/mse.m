function e = mse(Y, P)
  e = mean(mean((Y-P).^2));
end