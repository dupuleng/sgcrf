function l = log_loss(X, Y, Lambda, Theta, Sigma)
  m = size(X, 1);

  v = eig(Lambda);
  if min(v) < 0
    l = Inf;
    return
  end
  log_det_Lambda = sum(log(v));
  
  S = 1/m*[Y X]'*[Y X];
  Z = [Lambda Theta'; Theta Theta*Sigma*Theta'];
  
  l = -log_det_Lambda + trace(S*Z);
  