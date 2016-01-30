function [D df] = owl_qn_dir(X, G, L, Xh, Gh, rho)
memory = size(Xh, 3);

% Take the pseudo gradient
z = X == 0;
V = zeros(size(X));
V(~z) = G(~z) + L(~z).*sign(X(~z));
V(z) = st(G(z), L(z));
V = -V;

% Scale it by the pseudo inverse Hessian
D = V;
alpha = zeros(memory,1);
for i=1:memory
  if isnan(Gh(1,1,i))
    i = i - 1;
    break
  end
  
  alpha(i) = -rho(i)*trace(Xh(:,:,i)'*D);
  D = D + alpha(i)*Gh(:,:,i);
end
memory = i;

if memory > 0
  D = D * rho(1)/norm(Gh(:,:,1), 'fro');
  for i=memory:-1:1
    beta = rho(i)*trace(Gh(:,:,i)'*D);
    D = D - Xh(:,:,i)*(alpha(i) + beta);
  end
end

% Project it onto the same orthant
D(D .* V <= 0) = 0;

% Linear change from this direction
df = trace(V'*D);
