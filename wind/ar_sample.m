function S = ar_sample(model, N, U, x0, k)
n = size(U, 2);
p = size(model.NoiseVariance,1);

A = [zeros(k-1,1) eye(k-1); zeros(1, k-3) -model.a(4:-1:2)];
B = [zeros(k-1,71); cell2mat(model.b)];
C = [zeros(k-1,3); model.c(3:-1:1)];

epsilon = randn(k+2, N)*sqrt(model.NoiseVariance);
S = repmat(x0,1,N); 
for i=1:48
  S = A*S + repmat(B*U(i,:)',1,N) + C*epsilon(i:i+2,:);
end
S = S';
