
m = 50;
n = 200;
p = 50;

randn('seed', 0);
rand('seed', 0);

[Lambda0 Theta0] = gen_params3(n, p);
[X Y] = gen_data(Lambda0, Theta0, m*2);

t = 1:m;
s = m+1:2*m;

% Normalize columns and give Y's mean zero
X = bsxfun(@rdivide, X, sqrt(diag(X'*X)/m)');
Y = bsxfun(@minus, Y, mean(Y));

St = [Y(t,:) X(t,:)]'*[Y(t,:) X(t,:)]/m;

clear cv;
lambdas = logspace(1, -3, 20);
for i=1:length(lambdas)
  for j=1:length(lambdas)
    [Lambda Theta] = gcrf_newton(St, lambdas(i), lambdas(j), eye(p), zeros(n,p), ...
                                 params); 
    Sigma = inv(Lambda);
    P  = -X*Theta*Sigma;
    cv.mse_train(i,j) = mse(P(t,:), Y(t,:));
    cv.mse_test(i,j) = mse(P(s,:), Y(s,:));
    cv.ll_train(i,j) = log_loss(X(t,:), Y(t,:), Lambda, Theta, Sigma);
    cv.ll_test(i,j) = log_loss(X(s,:), Y(s,:), Lambda, Theta, Sigma);
    
    fprintf('%d\t%d\t%f\n', i, j, cv.mse_test(i,j));
  end
end

