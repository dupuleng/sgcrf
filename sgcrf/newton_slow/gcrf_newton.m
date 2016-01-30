function [Lambda Theta h] = gcrf_newton(X, Y, lambda, params)
[m n] = size(X); p = size(Y, 2);

if ~isfield(params, 'quiet') params.quiet = 1; end;
if ~isfield(params, 'max_iters') params.max_iters = 100; end;
if ~isfield(params, 'sigma') params.sigma = 0.001; end;
if ~isfield(params, 'beta')  params.beta = 0.5; end;
if ~isfield(params, 'epsilon') params.epsilon = 1e-2; end;
if ~isfield(params, 'd_tol') params.d_tol = 0.05; end;

S = cov([Y X]);
Syy = S(1:p,1:p);
Syx = S(1:p,p+1:p+n);
Sxx = S(p+1:p+n,p+1:p+n);

Lambda = eye(p);
Theta = zeros(n,p);
Sigma = eye(p);
R = eye(p);
f = objective(S, lambda, Lambda, Theta, R, Sigma);
 
tic;
for t=1:params.max_iters
  fprintf('Newton iteration %d\n', t);
  
  % Find the Newton direction
  clear dir_params;
  dir_params.max_iters = floor(1 + t/3);
  dir_params.d_tol = params.d_tol;
  dir_params.epsilon = params.epsilon;
  [U V] = argmin_dir(S, lambda, Lambda, Theta, Sigma, dir_params);
  
  % Find step size w/ backtracking line search
  df = trace(Syy*U) - trace(Sigma*U) - trace(Sigma*Theta'*Sxx*Theta*Sigma*U) ...
       + 2*trace(Syx*V) + trace(Sigma*Theta'*Sxx*V);
  dl1 = norm(vec(Theta+V),1) - norm(vec(Theta),1) + ...
        norm_offd(Lambda+U) - norm_offd(Lambda);

  alpha = 1/params.beta;  
  while true
    alpha = alpha*params.beta;
    [R err] = chol(Lambda + alpha*U);
    if err 
      continue
    end
    
    Sigma = inv(R)*inv(R)';
    f_alpha = objective(S, lambda, Lambda+alpha*U, Theta+alpha*V, R, Sigma);
    fprintf('  alpha=%f\tf=%f\n', alpha, f_alpha);
    
    if f_alpha <= f + alpha*params.sigma*(df + lambda*dl1)
      break
    end
  end

  Lambda = Lambda + alpha*U;
  Theta = Theta + alpha*V;
  f = f_alpha;
  
  h.objval(t) = f;
  h.time(t) = toc;
end