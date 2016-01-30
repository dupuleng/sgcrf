
n = 250;    % features 
p = 250;      % outputs 

% random data

[Theta0 Lambda0] = gen_params(n, p);

clear params;
params.quiet = 0;
params.max_iter = 1000;

lambda2 = 1e-4;
ms = 100:50:1000
for i=1:length(ms)
    m = ms(i)

    % training/validation data
    [Xt Yt] = gen_data(Theta0, Lambda0, m);
    [Xv Yv] = gen_data(Theta0, Lambda0, m);

    % pick best lambda1/lambda2 by cross validation
    clear cv;
    lambdas = logspace(1, -3, 10);
    
    clear params;
    params.quiet = 0;
    params.max_iter = 1000;
    for j=1:length(lambdas)
        lambda = [lambdas(j) lambda2]
        [Theta Lambda] = gcrf_admm(Xt, Yt, lambda, 1, params);
        
        Sigma = inv(Lambda);
        cv_size.cv(i).l1(j, :) = [mse_gcrf(Xt, Yt, Theta, Sigma)
                                  mse_gcrf(Xv, Yv, Theta, Sigma)];
    end
    [x k] = min(cv_size.cv(i).l1(:, 2));
    cv_size.lambda1(i) = lambdas(k);
    
    lambdas = logspace(5, -5, 50);
    for j = 1:length(lambdas)
        [Theta Lambda] = gcrf_ls(Xt, Yt, [0 lambdas(j)]);

        Sigma = inv(Lambda);
        cv_size.cv(i).l2(j, :) = [mse_gcrf(Xt, Yt, Theta, Sigma)
                                  mse_gcrf(Xv, Yv, Theta, Sigma)];
    end
    [x k] = min(cv_size.cv(i).l2(:, 2));
    cv_size.lambda2(i) = lambdas(k);

    % test data
    [Xs Ys] = gen_data(Theta0, Lambda0, m);

    [Theta1 Lambda1] = gcrf_admm(Xt, Yt, [cv_size.lambda1(i) lambda2], 1, params);
    [Theta2 Lambda2] = gcrf_ls(Xt, Yt, [0 cv_size.lambda2(i)]);
    Z = covsel([Yt Xt], cv_size.lambda1(i), 1, 1);
    Sigma = inv(Z);
    Theta = Sigma(p+1:p+n,p+1:p+n) \ Sigma(p+1:p+n,1:p);
  
    cv_size.loss(i, :) = [
        mse_gcrf(Xt, Yt, Theta1, inv(Lambda1))
        mse_gcrf(Xv, Yv, Theta1, inv(Lambda1))
        mse_gcrf(Xs, Ys, Theta1, inv(Lambda1))
        mse_gcrf(Xt, Yt, Theta2, inv(Lambda2))
        mse_gcrf(Xv, Yv, Theta2, inv(Lambda2))
        mse_gcrf(Xs, Ys, Theta2, inv(Lambda2))
        mse(Xt*Theta, Yt) 
        mse(Xv*Theta, Yv)
        mse(Xs*Theta, Ys)];

    cv_size.loss(i, :)

    % checkpoint
    save('cv_size.mat', 'cv_size');
end


