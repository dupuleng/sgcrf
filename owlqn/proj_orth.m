function y = proj_orth(x, d)
y = x+d;
y(y.*x < 0) = 0;