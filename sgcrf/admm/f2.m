function l = f2(x, x0, p, S, rho)
if any(isnan(x))
  l = Inf;
  return;
end

Z = reshape(x, [], p);
Lambda = Z(1:p,:);
Theta = Z(p+1:end,:);

det_Lambda = det(Lambda);
if det_Lambda <= 0
  l = Inf;
  return;
end

Z2 = [Lambda Theta'; Theta Theta/Lambda*Theta'];
l = -log(det_Lambda) + trace(S*Z2) + rho/2*sum((x-x0).^2);
