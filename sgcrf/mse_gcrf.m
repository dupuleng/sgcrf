function e = mse_gcrf(X, Y, Theta, Sigma)
  P = -X*Theta*Sigma;
  e = mean(mean((Y-P).^2));
end