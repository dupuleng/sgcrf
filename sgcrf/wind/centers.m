function mus = centers(X, d)
[m n] = size(X);
mus = zeros(d,n);
mus(1,:) = X(randi(m), :);
for i=2:d  
  D = sqdist(X', mus(1:i-1,:)');
  D = min(D, [], 2);
  mus(i,:) = X(randsample(m,1,true,D), :);
end