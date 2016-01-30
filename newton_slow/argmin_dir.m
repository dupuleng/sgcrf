function [U V] = argmin_dir(S, lambda, Lambda, Theta, Sigma, params)
[n p] = size(Theta);
Syy = S(1:p,1:p);
Syx = S(1:p,p+1:p+n);
Sxy = S(p+1:p+n,1:p);
Sxx = S(p+1:p+n,p+1:p+n);

if ~isfield(params, 'd_tol') params.d_tol = 0.05; end;
if ~isfield(params, 'epsilon') params.epsilon = 1e-2; end;
if ~isfield(params, 'max_iters') params.max_iters = 100; end;

% Free set - off-diagonal of Lambda
G = Syy - Sigma - Sigma*Theta'*Sxx*Theta*Sigma;
[Lambda_is Lambda_js] = find(...
    triu(Lambda ~= 0 | 2*abs(G) >= lambda + params.epsilon, 1));

% Free set - Theta
G = 2*Syx' + 2*Sxx*Theta*Sigma;
[Theta_is Theta_js] = find(Theta ~= 0 | abs(G) >= lambda + params.epsilon);  

fprintf('  active set size: %d\n', length(Lambda_is) + length(Theta_is) + p);

A = Sigma*Theta'*Sxx*Theta*Sigma;
B = Sigma*Theta'*Sxx;
C = Sxx*Theta*Sigma;

U = zeros(p);
V = zeros(n,p);
USigma = zeros(p);
VSigma = zeros(n,p);

for t=1:params.max_iters
  Uold = U;
  Vold = V;

  % Minimize over diagonal elements of Lambda
  for i = 1:p
    a = Sigma(i,i).^2 + 2*Sigma(i,i)*A(i,i);
    b = -Sigma(i,i) + Syy(i,i) - A(i,i) + Sigma(i,:)*USigma(:,i) ...
        - 2*B(i,:)*VSigma(:,i) + 2*A(i,:)*USigma(:,i);
    mu = -b/a;
    U(i,i) = U(i,i) + mu;
    USigma(i,:) = USigma(i,:) + mu*Sigma(i,:);
  end
  
  % Minimize over off-diagonal elements of Lambda
  for k=1:numel(Lambda_is)
    i = Lambda_is(k); j = Lambda_js(k);

    a = Sigma(i,j).^2 + Sigma(i,i)*Sigma(j,j) + Sigma(i,i)*A(j,j) + ...
        2*Sigma(i,j)*A(i,j) + Sigma(j,j)*A(i,i);
    b = -Sigma(i,j) + Syy(i,j) - A(i,j) + Sigma(i,:)*USigma(:,j) ...
        - B(i,:)*VSigma(:,j) - B(j,:)*VSigma(:,i) ...
        + A(i,:)*USigma(:,j) + A(j,:)*USigma(:,i);
    c = Lambda(i,j) + U(i,j);  

    mu = -c + st(c - b/a, lambda/a);
    U(i,j) = U(i,j) + mu;
    U(j,i) = U(j,i) + mu;
    USigma(i,:) = USigma(i,:) + mu*Sigma(j,:);
    USigma(j,:) = USigma(j,:) + mu*Sigma(i,:);
  end
  
  % Minimize over each element of Theta
  for k = 1:numel(Theta_is)
    i = Theta_is(k); j = Theta_js(k);
    
    a = 2*Sigma(j,j)*Sxx(i,i);
    b = 2*Sxy(i,j) + 2*C(i,j) + 2*Sxx(i,:)*VSigma(:,j) - 2*C(i,:)*USigma(:,j);
    c = Theta(i,j) + V(i,j);
    
    mu = -c + st(c - b/a, lambda/a);
    V(i,j) = V(i,j) + mu;
    VSigma(i,:) = VSigma(i,:) + mu*Sigma(j,:);
  end

  normU = norm(U, 'fro')/p;
  normV = norm(V, 'fro')/sqrt(n*p);
  diffU = norm(U - Uold, 'fro')/p;
  diffV = norm(V - Vold, 'fro')/sqrt(n*p);

  if diffU < normU*params.d_tol && (normV == 0 || diffV < normV*params.d_tol)
    break
  end
  
  fprintf('  coordinate descent, normU=%f\tdiffU=%f\tnormV=%f\tdiffV=%f\n', ...
          normU, diffU, normV, diffV);
end



