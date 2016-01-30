function [Lambda Theta h] = gcrf_fista(S, p, lambda, params)
n = size(S,1) - p;

if ~isfield(params, 'max_iters') params.max_iters = 100; end;
if ~isfield(params, 'step') params.step = 0.01; end;
if ~isfield(params, 'epsilon') params.epsilon = 1e-2; end;

Syy = S(1:p,1:p);
Syx = S(1:p,p+1:p+n);
Sxy = S(p+1:p+n,1:p);
Sxx = S(p+1:p+n,p+1:p+n);

Lambda_p = eye(p);
Lambda = eye(p);
Sigma = eye(p);
Theta_p = zeros(n,p);
Theta = zeros(n,p);

tic;
for t=1:params.max_iters
  Lambda_y = Lambda + (t-2)/(t+1)*(Lambda - Lambda_p);
  [R err] = chol(Lambda_y);
  assert(~err, 'Acceleration point not PSD');
  
  Sigma = inv(R)*inv(R)';
  g_Lambda = -Sigma + Syy - Sigma*Theta'*Sxx*Theta*Sigma;
  Lambda0 = Lambda_y - params.step*g_Lambda;
  D = diag(diag(Lambda0));
  Lambda0 = st(Lambda0 - D, lambda*params.step) + D;
  
  [R err] = chol(Lambda0);
  assert(~err, 'Gradient step not PSD');
  Lambda_p = Lambda;
  Lambda = Lambda0;  
  Sigma = inv(R)*inv(R)';

  Theta_y = Theta + (t-2)/(t+1)*(Theta - Theta_p);
  g_Theta = 2*Sxy + 2*Sxx*Theta_y*Sigma;
  Theta_p = Theta;
  Theta = st(Theta_y - params.step*g_Theta, lambda*params.step);
  
  h.objval(t) = objective(S, lambda, Lambda, Theta, R, Sigma);
  h.time(t) = toc;

  fprintf('%d\tobjval=%f\n', t, h.objval(t));
  
  if max(max([g_Lambda; g_Theta])) < params.epsilon
    break
  end
end