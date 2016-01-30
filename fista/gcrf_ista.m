function [Lambda Theta h] = gcrf_ista(X, Y, lambda, params)
[m n] = size(X); p = size(Y,2);

if ~isfield(params, 'max_iters') params.max_iters = 100; end;
if ~isfield(params, 'step') params.step = 0.01; end;
if ~isfield(params, 'epsilon') params.epsilon = 1e-2; end;

S = cov([Y X]);
Syy = S(1:p,1:p);
Syx = S(1:p,p+1:p+n);
Sxy = S(p+1:p+n,1:p);
Sxx = S(p+1:p+n,p+1:p+n);

Lambda = eye(p);
Sigma = eye(p);
Theta = zeros(n,p);

tic;
for t=1:params.max_iters
  g_Lambda = -Sigma + Syy - Sigma*Theta'*Sxx*Theta*Sigma;
  Lambda0 = Lambda - params.step*g_Lambda;
  D = diag(diag(Lambda0));
  Lambda0 = st(Lambda0 - D, lambda*params.step) + D;
  [R err] = chol(Lambda0);
  if err
    fprintf('Gradient step not PSD');
    break
  end
  Lambda = Lambda0;  
  Sigma = inv(R)*inv(R)';
  
  g_Theta = 2*Sxy + 2*Sxx*Theta*Sigma;
  Theta = st(Theta - params.step*g_Theta, lambda*params.step);
  
  h.objval(t) = objective(S, lambda, Lambda, Theta, R, Sigma);
  h.time(t) = toc;
  
  fprintf('%d\tobjval=%f\t||Theta||_0=%f\t||Lambda||_0=%f\n', t, ...
          h.objval(t), nnz(Theta)/numel(Theta), nnz(Lambda)/numel(Lambda));
  
  if max(max([g_Lambda; g_Theta])) < params.epsilon
    break
  end
end