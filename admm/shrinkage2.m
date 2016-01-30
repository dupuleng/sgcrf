function Z = shrinkage2(Z0, k)    
  p = size(Z0,2);
  n = size(Z0,1) - p;
  Lambda0 = Z0(1:p,:);
  Theta0 = Z0(p+1:end,:);
  Z =  [max(0, Lambda0-k) - max(0, -Lambda0-k);
        max(0, Theta0-2*k) - max(0, -Theta0-2*k)];
  % first p diagonal elements
  m = 1:n+p+1:p*(n+p);
  Z(m) = Z0(m);
end


    
