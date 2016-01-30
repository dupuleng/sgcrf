function [Lambda Theta h] = gcrf_owlqn(S, p, lambda, params)
n = size(S,1) - p;

if ~isfield(params, 'max_iters') params.max_iters = 100; end;
if ~isfield(params, 'memory') params.memory = 10; end;
if ~isfield(params, 'sigma') params.sigma = 0.001; end;
if ~isfield(params, 'beta') params.beta = 0.5; end;
if ~isfield(params, 'qn') params.qn = 1; end;

Syy = S(1:p,1:p);
Syx = S(1:p,p+1:p+n);
Sxy = S(p+1:p+n,1:p);
Sxx = S(p+1:p+n,p+1:p+n);

Lambda = eye(p);
Sigma = eye(p);
R = eye(p);
Theta = zeros(n,p);
f = objective(S, lambda, Lambda, Theta, R, Sigma);

% History of X and G used for L-BFGS
Xh = nan(n+p, p, params.memory);
Gh = nan(n+p, p, params.memory);
rho = nan(params.memory, 1);

% L1 penalty - don't penalize the diagonal elements of Lambda
L = lambda*[ones(p)-eye(p); ones(n,p)];

tic;
for t=1:params.max_iters 
  fprintf('OWL-QN iteration %d\n', t);  
  
  G = [-Sigma + Syy - Sigma*Theta'*Sxx*Theta*Sigma
       2*Sxy + 2*Sxx*Theta*Sigma];
  
  % bookkeeping for L-BFGS
  if params.qn && t ~= 1
    Xh(:, :, 2:params.memory) = Xh(:, :, 1:params.memory-1);
    Xh(:, :, 1) = [Lambda; Theta] - X_prev;
    Gh(:, :, 2:params.memory) = Gh(:, :, 1:params.memory-1);
    Gh(:, :, 1) = G - G_prev;
    rho(2:params.memory) = rho(1:params.memory-1);
    rho(1) = 1 / trace(Xh(:,:,1)'*Gh(:,:,1));
  end
  
  [D df] = owl_qn_dir([Lambda; Theta], G, L, Xh, Gh, rho);
    
  % backtracking line search
  alpha = 1/params.beta;
  while true
    alpha = alpha*params.beta;
    Lambda_alpha = proj_orth(Lambda, alpha*D(1:p,:));
    Theta_alpha = proj_orth(Theta, alpha*D(p+1:end,:));

    [R err] = chol(Lambda_alpha);
    if err
      continue
    end
      
    Sigma = inv(R)*inv(R)';
    f_alpha = objective(S, lambda, Lambda_alpha, Theta_alpha, R, Sigma);
    fprintf('  alpha=%f\tf=%f\n', alpha, f_alpha);
    
    if f_alpha <= f - alpha*params.sigma*df
      break
    end
  end
  
  X_prev = [Lambda; Theta];
  G_prev = G;

  Lambda = Lambda_alpha;
  Theta = Theta_alpha;
  f = f_alpha;
  
  h.objval(t) = f;
  h.time(t) = toc;
end