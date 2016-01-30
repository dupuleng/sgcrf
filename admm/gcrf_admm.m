function [Theta Lambda history] = gcrf_admm(X, Y, lambda, rho, params)
  [m n] = size(X); p = size(Y, 2);  

  if ~isfield(params, 'quiet') params.quiet = 1; end;
  if ~isfield(params, 'max_iter') params.max_iter = 100; end;
  if ~isfield(params, 'abstol') params.abstol = 1e-6; end;
  if ~isfield(params, 'reltol')   params.reltol = 1e-4; end;
  
  Z = {}; Z{1} = zeros(n+p); Z{2} = Z{1}; Z{3} = Z{1};
  V = {}; V{1} = zeros(n+p); V{2} = V{1};
  S = [Y X]'*[Y X]/m + lambda(2)*eye(n+p);

  tic;
  for k = 1:params.max_iter
    Zold = Z;

    Z{1} = argmin_f((Z{2}+Z{3}+V{1})/2 - S/rho, p, rho);
    Z{2} = shrinkage((Z{1}+Z{3}+V{2})/2, p, lambda(1)/rho);
    Z{3} = proj_sd((Z{1}+Z{2}-V{1}-V{2})/2);

    V{1} = V{1}+Z{2}+Z{3}-2*Z{1};
    V{2} = V{2}+Z{1}+Z{3}-2*Z{2};
    
    history.r_norm(k,:) = [
        norm(Z{1}-Z{2}, 'fro') ...
        norm(Z{2}-Z{3}, 'fro') ...
        norm(Z{3}-Z{1}, 'fro')];
    history.s_norm(k,:) = [
        rho*(norm((Zold{1}-Z{1})+(Zold{3}-Z{3}), 'fro')) ...
        rho*norm((Zold{3}-Z{3}), 'fro')];
            
    history.eps_pri(k) = (p+n)*params.abstol + params.reltol*max([
        norm(Z{1}, 'fro') norm(Z{2}, 'fro') norm(Z{3}, 'fro')]);
    history.eps_dual(k) = (p+n)*params.abstol + params.reltol*max([
        norm(V{1}, 'fro') norm(V{2}, 'fro')]);
        
    Theta = Z{3}(p+1:p+n, 1:p);
    Lambda = Z{3}(1:p, 1:p);
    history.objval(k) = admm_obj(S, Z, p, lambda, rho);
    history.time(k) = toc;

    if ~params.quiet
      fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.4f\n', k, ...
              history.r_norm(k), history.eps_pri(k), ...
              history.s_norm(k), history.eps_dual(k), ...
              history.objval(k));
    end
    
    if (max(history.r_norm(k,:)) < history.eps_pri(k) && ...
        max(history.s_norm(k,:)) < history.eps_dual(k))
      break
    end
    
  end
end

