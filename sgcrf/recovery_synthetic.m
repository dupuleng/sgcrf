
randn('seed', 1);
rand('seed', 1);

clear params;
params.max_iters = 100;
params.tol = 1e-4;
params.quiet = 1;

% Recovery for the CRF chain graph
ps = [50 100 200 400];
ms = floor(linspace(500, 5000, 10));
Np = length(ps);
Nm = length(ms);
Nt = 20;

% Generate a single X to use for all examples
m = ms(Nm);
n = ps(Np);

%X = randn(m,n);
%X(:, 1:2:n) = X(:, 1:2:end).^2;
X = zeros(m, n);

success = zeros(Np, Nm);
for k=1:Np
  p = ps(k);
  n = p;

  % % Chain graph
  % Lambda0 = 0.2*diag(ones(p-1,1),1) + 0.2*diag(ones(p-1,1),-1) + eye(p);
  % Theta0 = 0.2*diag(ones(p, 1));
 
  % CRF graph
  Omega = zeros(n+p);
  Omega(1:p,1:p) = 0.2*diag(ones(p-1,1),1);
  Omega(1:p,p+1:p+n) = 0.2*diag(ones(p,1));
  Omega = Omega + Omega' + eye(n+p);

  Lambda0 = Omega(1:p,1:p);
  Sigma0 = inv(Lambda0);
  Theta0 = Omega(p+1:p+n,1:p);

  for i = 1:Nm
    m = ms(i);
    lambda = 4*sqrt((log(p))/m);
    
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
      
      fprintf('p=%d\tm=%d\tsuccess=%f\t%d/%d\t%d/%d\n', p, m, sum(success_i)/j, ...
              l_fp, l_fn, t_fp, t_fn);
    end
    success(k, i) = sum(success_i)/Nt;
  end
end

save('recovery_synthetic', 'ms', 'ps', 'success');

figure;
plot(ms, success, 'LineWidth', 1)
legend('p=50', 'p=100', 'p=200', 'p=400', 'Location', 'SouthEast');

figure;
plot((repmat(ms,Np,1)'./repmat(log(ps),Nm,1)), success', 'LineWidth', 1)
legend('p=50', 'p=100', 'p=200', 'p=400', 'Location', 'SouthEast');
