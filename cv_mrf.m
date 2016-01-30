function cv = cv_mrf(Xt, Yt, Xs, Ys, lambdas, tol, max_iters)
[m n] = size(Xt); p = size(Yt,2);
L = ones(p+n) - diag(ones(p+n,1));
St = [Yt Xt]'*[Yt Xt]/m;

[Zs Zinvs opt time iter dGap] = QUIC('path', St, L, lambdas, tol, 2, max_iters);

for i=1:length(lambdas)
  Z = Zs(:,:,i);
  Zinv = Zinvs(:,:,i);
  Theta = Zinv(p+1:p+n,p+1:p+n) \ Zinv(p+1:p+n,1:p);
  cv.mse(i,:) = [mse(Xt*Theta, Yt) mse(Xs*Theta, Ys)];
end

