function Z = argmin_f2(Z0, S, rho)
p = size(Z0,2);
x0 = Z0(:);
f = @(x) f2(x, x0, p, S, rho);
g = @(x) grad_f2(x, x0, p, S, rho);
l = -inf(numel(x0),1);
u = inf(numel(x0),1);
opts = struct('x0', x0);

% t = 1e-2;
% x = x0;
% for i=1:100
%   gx = g(x);
%   fx = f(x-t*gx);
%   if isinf(fx)
%     break
%   end
%   fprintf('%d\t%10.4f\t%10.4f\n', i, norm(gx), fx);
%   x = x-t*gx;
% end

[x fx info] = lbfgsb({f,g}, l, u, opts);
Z = reshape(x, size(Z0));


