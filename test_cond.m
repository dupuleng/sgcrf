
n = 10;  % features 
p = 5;  % outputs 
m = 20;  % examples

lambda = [0 0 0 0];

clear run;
for k=1:100
  % random data
  Theta0 = rand_sparse(n, p, 0.3);
  Lambda0 = rand_precision(p, 0.3);  
  Phi = randn(m, n);
  Y = gcrf_sample(Phi, Lambda0, Theta0);
  run.cond(k) = cond([Phi Y]);

  [Theta Lambda] = gcrf_cvx(Phi, Y, lambda);
  run.loss_cvx(k) = objective(Phi, Y, Theta, Lambda, inv(Lambda), lambda);
    
  [Theta_a Lambda_a history] = gcrf_admm(Phi, Y, lambda); 
  Run.loss_admm(k) = objective(Phi, Y, Theta_a, Lambda_a, inv(Lambda_a), ...
                               lambda);
  
  run.loss_admm(k)
  
  if run.loss_admm(k) - run.loss_cvx(k) > 10
    break
  end
end

