function obj = admm_obj(S, Z, p, lambda, rho)
  obj = trace(Z{1}*S) - log(det(Z{1}(1:p,1:p))) + ...
        2*lambda(1)*norm(vec(Z{2}(p+1:end,1:p)),1) + ...
        2*lambda(1)*norm(vec(triu(Z{2}(1:p,1:p),1)),1);
  
  
  
