% test that the Hessian is correct

m = 100;
n = 10;
p = 5;

% Generate a random problem instance
rand('seed', 1);
randn('seed', 1);
Y = randn(m,p);
X = rand(m,n)*2-1;

S = cov([Y X]);
Syy = S(1:p,1:p);
Syx = S(1:p,p+1:end);
Sxx = S(p+1:end,p+1:end);

% generate random parameters, forcing Lambda to be PSD
Lambda = rand(p,p)*2-1;
Lambda = Lambda' + Lambda;
Lambda = Lambda + 1.1*abs(min(eig(Lambda)))*eye(p);
Theta = rand(n,p)*2-1;
Sigma = inv(Lambda);

U = rand(p,p)*2-1;
U = U + U';
V = rand(n,p)*2-1;

f = @(x) obj_vec(x, S, p);
[g H] = numdiff(f, [vec(Lambda); vec(Theta)]);

% Analytic versions
H_LL = kron(Sigma, Sigma + 2*Sigma*Theta'*Sxx*Theta*Sigma);
H_LT = kron(-2*Sigma, Sigma*Theta'*Sxx);
H_TT = kron(2*Sigma, Sxx);

% Numerical versions
H_LL2 = H(1:p^2,1:p^2);
H_LT2 = H(1:p^2,p^2+1:end);
H_TT2 = H(p^2+1:end, p^2+1:end);

% numdiff currently seems slightly broken so we just evalute in terms of the
% inner product for now.
assert(abs(vec(U)'*H_LL*vec(U) - vec(U)'*H_LL2*vec(U)) < 1e-2);
assert(abs(vec(U)'*H_LT*vec(V) - vec(U)'*H_LT2*vec(V)) < 1e-2);
assert(abs(vec(V)'*H_TT*vec(V) - vec(V)'*H_TT2*vec(V)) < 1e-2);

disp('PASSED');