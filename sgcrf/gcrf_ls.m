function [Lambda Theta] = gcrf_ls(S, p)
  Syy = S(1:p, 1:p);
  Sxx = S(p+1:end, p+1:end);
  Sxy = S(p+1:end, 1:p);
  Lambda = inv(Syy - Sxy'/Sxx*Sxy);
  Theta = -Sxx\Sxy*Lambda;

  