function cv = cv_ls_copula(Xt, Yt, Xs, Ys, lambdas)
[m p] = size(Yt);

Y_norm = copula(Yt);
St = [Y_norm Xt]'*[Y_norm Xt]/m;
for i=1:length(lambdas)
  [Lambda Theta] = gcrf_ls(St + eye(size(St,1))*lambdas(i), p);
  Pt_norm  = -Xt*Theta / Lambda;
  Ps_norm  = -Xs*Theta / Lambda;
  Pt = inv_copula(Yt, Pt_norm);
  Ps = inv_copula(Yt, Ps_norm);
  cv.mse(i,:) = [mse(Pt, Yt) mse(Ps, Ys)];
  cv.ll(i,:) = [log_loss(Xt, Yt, Lambda, Theta, inv(Lambda))
                log_loss(Xs, Ys, Lambda, Theta, inv(Lambda))];

  fprintf('%d\t%f\t%f\t%f\t%f\n', i, cv.ll(i,:), cv.mse(i,:));

  if i == 1 || cv.ll(i,2) < cv.ll(i-1,2)
    cv.Lambda = Lambda;
    cv.Theta = Theta;
  end
end

  
  