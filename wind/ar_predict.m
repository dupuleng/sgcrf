
function x = ar_predict(model, U, x0, k)
n = size(U, 2);
A = [zeros(k-1,1) eye(k-1); zeros(1, k-3) -model.a(4:-1:2)];
B = [zeros(k-1,n); cell2mat(model.b)];

x = x0;
for i=1:k
  x = A*x + B*U(i,:)';
end




