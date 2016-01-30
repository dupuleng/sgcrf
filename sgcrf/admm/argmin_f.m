function Z = argmin_f(Z, p, rho)
  [Q D] = eig(rho*Z(1:p, 1:p));
  es = diag(D);
  Z(1:p, 1:p) = Q * diag((es + sqrt(es.^2+4*rho))/(2*rho)) * Q';
  Z = (Z+Z')/2;  
end
