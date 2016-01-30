function c = check_coverage(s, y, alphas) 
x = sort(s);
c = zeros(length(alphas),1);
for i=1:length(alphas)
  idx = ceil(length(s)*alphas(i)/2);
  c(i) = x(idx) < y && x(end-idx) > y;
end

