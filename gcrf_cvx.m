function [Theta Lambda] = gcrf_cvx(X, Y, lambda)
  [m n] = size(X); p = size(Y, 2);  
  S = 1/m*[Y X]'*[Y X] + lambda(2)*eye(n+p);

  cvx_begin
    variable Z(p+n,p+n) symmetric;
    minimize( -log_det(Z(1:p, 1:p)) + trace(Z*S) + ...
              lambda(1)*sum(sum(abs(Z(1:p, 1:p) - diag(diag(Z(1:p, 1:p)))))) ...
              + lambda(1)*sum(sum(abs(Z(p+1:p+n, 1:p)))) );
    subject to
      Z == semidefinite(p+n);
  cvx_end
  
  Theta = Z(p+1:p+n, 1:p);
  Lambda = Z(1:p,1:p);
end
