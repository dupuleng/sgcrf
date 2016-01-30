function obj = admm2_obj(S, X, Z, p, lambda)
  det_Lambda = det(X(1:p,:));
  if det_Lambda <= 0
    obj = Inf;
    return
  end
  
  X2 = [X(1:p,:) X(p+1:end,:)'
        X(p+1:end, :) X(p+1:end, :)/X(1:p,:)*X(p+1:end,:)'];
  obj = -log(det_Lambda) + trace(S*X2) + ...
        2*lambda(1)*norm(vec(Z(p+1:end,:)),1) + ...
        2*lambda(1)*norm(vec(triu(Z(1:p,:),1)),1);
end  
  