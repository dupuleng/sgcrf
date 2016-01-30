function [Theta Lambda history] = gcrf_admm2(X, Y, lambda, rho, params)
  [m n] = size(X); p = size(Y, 2);  

  if ~isfield(params, 'quiet') params.quiet = 1; end;
  if ~isfield(params, 'max_iter') params.max_iter = 100; end;
  if ~isfield(params, 'abstol') params.abstol = 1e-4; end;
  if ~isfield(params, 'reltol')   params.reltol = 1e-2; end;

  S = [Y X]'*[Y X]/m + lambda(2)*eye(n+p);
  X = [eye(p); zeros(n,p)];
  Z = [eye(p); zeros(n,p)];
  U = zeros(n+p,p);

  tic;
  for k = 1:params.max_iter
    X = argmin_f2(Z - U, S, rho);
    Zold = Z;
    Z = shrinkage2(X + U, lambda(1)/rho);
    U = U + X - Z;
        
    history.objval(k) = admm2_obj(S, X, Z, p, lambda);

    history.r_norm(k) = norm(X - Z, 'fro');
    history.s_norm(k) = norm(rho*(Z -Zold), 'fro');
                
    history.eps_pri(k) = sqrt(p*p+p*n)*params.abstol + ...
        params.reltol*max([norm(X, 'fro') norm(Z, 'fro')]);
    history.eps_dual(k) = sqrt(p*p+p*n)*params.abstol + ...
        params.reltol*norm(rho*U,'fro');
    history.time(k) = toc;
    
    if ~params.quiet
      fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.4f\n', k, ...
              history.r_norm(k), history.eps_pri(k), ...
              history.s_norm(k), history.eps_dual(k), ...
              history.objval(k));
    end
    
    if (history.r_norm(k) < history.eps_pri(k) && ...
        history.s_norm(k) < history.eps_dual(k))
      break
    end
  
  end

Theta = Z(p+1:end,:);
Lambda = Z(1:p,:);

