
randn('seed', 1);
rand('seed', 1);

clear params;
params.max_iters = 100;
params.tol = 1e-4;
params.quiet = 1;

% Recovery for the CRF chain graph
ns = [50 100 200 400];
ms = floor(linspace(500, 5000, 10));
Nn = length(ns);
Nm = length(ms);
Nt = 20;
p = 50;

success = zeros(Nn, Nm);
for k=1:Nn
  n = ns(k);

  % % Chain graph
  % Lambda0 = 0.2*diag(ones(p-1,1),1) + 0.2*diag(ones(p-1,1),-1) + eye(p);
  % Theta0 = 0.2*diag(ones(p, 1));
 
  % CRF graph + noise
  Omega = zeros(n+p);
  Omega(1:p,1:p) = 0.2*diag(ones(p-1,1),1);
  Omega(1:p,p+1:p+p) = 0.2*diag(ones(p,1));
  Omega = Omega + Omega' + eye(n+p);

  Lambda0 = Omega(1:p,1:p);
  Sigma0 = inv(Lambda0);
  Theta0 = Omega(p+1:p+n,1:p);

  for i = 1:Nm
    m = ms(i);
    lambda = 3*sqrt((log(n))/m);
    
    success_i = zeros(Nt, 1);
    for j = 1:Nt
      Z = mvnrnd(zeros(m, n+p), inv(Omega));
      Y = Z(1:m, 1:p);
      X = 0.5*Z(1:m, p+1:p+n);
      S = [Y X]'*[Y X]/m;
      [Lambda Theta h0] = gcrf_newton(S, lambda, eye(p), zeros(n, p), ...
                                      params);
      assert(length(h0.objval) ~= params.max_iters);
      
      success_i(j) = (norm(sign(Lambda) - sign(Lambda0)) == 0 && ...
                      norm(sign(Theta) - sign(Theta0)) == 0);      
      l_fp = length(find(((Lambda~=0) - (Lambda0~=0)) == 1));
      l_fn = length(find(((Lambda0~=0) - (Lambda~=0)) == 1));
      t_fp = length(find(((Theta~=0) - (Theta0~=0)) == 1));
      t_fn = length(find(((Theta0~=0) - (Theta~=0)) == 1));
      
      fprintf('n=%d\tm=%d\tsuccess=%f\t%d/%d\t%d/%d\n', n, m, sum(success_i)/j, ...
              l_fp, l_fn, t_fp, t_fn);
    end
    success(k, i) = sum(success_i)/Nt;
  end
end

save('recovery_synthetic_noise', 'ms', 'ns', 'success');

figure;
plot(ms, success, 'LineWidth', 1)
legend('n=50', 'n=100', 'n=200', 'n=400', 'Location', 'SouthEast');

figure;
plot((repmat(ms,Nn,1)'./repmat(log(ns),Nm,1)), success', 'LineWidth', 1)
legend('n=50', 'n=100', 'n=200', 'n=400', 'Location', 'SouthEast');

