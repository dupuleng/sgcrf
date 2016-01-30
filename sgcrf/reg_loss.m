function l = reg_loss(Theta, Lambda, lambda)
  l = lambda(1)*sum(sum(abs(Theta))) + ...
      lambda(1)*sum(sum(abs(Lambda - diag(diag(Lambda))))) + ...
      lambda(2)*(trace(Lambda) + trace(Theta*inv(Lambda)*Theta'));
  
  

