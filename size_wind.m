setup_wind;

clear size;
size.ks = linspace(0.01, 1, 10);
size.lambdas_l2 = logspace(5,-5,50);
size.lambdas_l1 = logspace(1,-3,10);

clear params;
params.max_iters = 1000;
params.epsilon = 1e-2;

rand('state', 1);
p = randperm(length(t));

for k=1:length(size.ks)
  tk = t(p(1:floor(size.ks(k)*length(t))));
  size.cv_ls(k) = cv_ls(Phi(tk,:), Y(tk,:), Phi(s,:), Y(s,:), size.lambdas_l2);
  size.cv_crf(k) = cv_crf(Phi(tk,:), Y(tk,:), Phi(s,:), Y(s,:), size.lambdas_l1, params);
end

save('size_wind', 'size', 'params');