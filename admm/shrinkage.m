function Z = shrinkage(Z0, p, k)  
  Z = max(0, Z0-k) - max(0, -Z0-k);
  % first p diagonal elements
  m = 1:length(Z0)+1:p*length(Z0);
  Z(m) = Z0(m);
  Z(p+1:end,p+1:end) = Z0(p+1:end,p+1:end);
  Z = (Z+Z')/2;  
end


    
