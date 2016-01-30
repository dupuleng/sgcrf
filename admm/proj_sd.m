function Z = proj_sd(Z)
  [V L] = eig(Z);
  Z = V*max(real(L),0)*V';
  Z = (Z+Z')/2;  
end
