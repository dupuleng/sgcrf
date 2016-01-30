function cv = cv_crf_copula(Xt, Yt, Xs, Ys, lambdas, params)
[m p] = size(Yt); n = size(Xt,2);

if ~isfield(params, 'stop_early') params.stop_early = 0; end;

Y_norm = copula(Yt);
St = [Y_norm Xt]'*[Y_norm Xt]/m;
Lambda0 = eye(p);
Theta0 = zeros(n,p);
for i=1:length(lambdas)
  [Lambda Theta] = gcrf_newton(St, lambdas(i), Lambda0, Theta0, params);
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

  if i == 1 || cv.ll(i,2) < cv.ll(i-1,2)
    cv.Lambda = Lambda;
    cv.Theta = Theta;
  end
  
  if i > 1 && params.stop_early && cv.ll(i,2) > cv.ll(i-1,2)
    break
  end

  Theta0 = Theta;
  Lambda0 = Lambda;
end

  
  