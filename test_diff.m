% basic tests for first/second differential of sparse GCRF objective

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

U = rand(p,p)*2-1;
U = U + U';
V = rand(n,p)*2-1;
Sigma = inv(Lambda);

% test Lambda gradients numerically
f = @(x) -log(det(reshape(x,[],p)));
g = numdiff(f, vec(Lambda));
g2 = vec(-inv(Lambda));
assert(abs(norm(g2-g)) < 1e-2);

f = @(x) trace(reshape(x,[],p)*S(1:p,1:p));
g = numdiff(f, vec(Lambda));
g2 = vec(S(1:p,1:p));
assert(abs(norm(g2-g)) < 1e-2);

f = @(x) trace(Theta/reshape(x,[],p)*Theta'*Sxx);
g = numdiff(f, vec(Lambda));
g2 = vec(-Lambda\Theta'*Sxx*Theta/Lambda);
assert(abs(norm(g2-g)) < 1e-2);

% test Theta gradients numerically
f = @(x) 2*trace(reshape(x,[],p)*Syx);
g = numdiff(f, vec(Theta));
g2 = vec(2*Syx');
assert(abs(norm(g2-g)) < 1e-2);

f = @(x) trace(reshape(x,[],p)/Lambda*reshape(x,[],p)'*Sxx);
g = numdiff(f, vec(Theta));
g2 = vec(2*Sxx*Theta/Lambda);
assert(abs(norm(g2-g)) < 1e-2);

% test Theta / Lambda gradients together
f = @(x) obj_vec(x, S, p);
g = numdiff(f, [vec(Lambda); vec(Theta)]);
g2_L = vec(-inv(Lambda) + S(1:p,1:p) - Lambda\Theta'*Sxx*Theta/Lambda);
g2_T = vec(2*Syx' + 2*Sxx*Theta/Lambda);
assert(norm(g2_L-g(1:p*p)) < 1e-2);
assert(norm(g2_T-g(p*p+1:end)) < 1e-2);

% test Theta / Lambda first differential
f = @(x) obj_vec(x, S, p);
g = numdiff(f, [vec(Lambda); vec(Theta)]);
df = -trace(Sigma*U) + trace(Syy*U) + 2*trace(Syx*V) + ...
     2*trace(Sigma*Theta'*Sxx*V) - trace(Sigma*Theta'*Sxx*Theta*Sigma*U);
assert(abs([vec(U); vec(V)]'*g - df) < 1e-2);

% test Lambda second differentials
f = @(x) -log(det(reshape(x,[],p)));
[g H] = numdiff(f, vec(Lambda));
d2f = trace(Sigma*U*Sigma*U);
assert(abs(vec(U)'*H*vec(U) - d2f) < 1e-2);

f = @(x) trace(Theta/reshape(x,[],p)*Theta'*Sxx);
[g H] = numdiff(f, vec(Lambda));
d2f = 2*trace(Sigma*U*Sigma*Theta'*Sxx*Theta*Sigma*U);
assert(abs(vec(U)'*H*vec(U) - d2f) < 1e-2);

% test Theta second differentials
f = @(x) trace(reshape(x,[],p)/Lambda*reshape(x,[],p)'*Sxx);
[g H] = numdiff(f, vec(Theta));
d2f = 2*trace(Sigma*V'*Sxx*V);
assert(abs(vec(V)'*H*vec(V) - d2f) < 1e-2);

% Test the cross term
f = @(x) 2*trace(reshape(x,[],p)\Theta'*Sxx*V);
g = numdiff(f, vec(Lambda));
g2 = vec(-2*Sigma*V'*Sxx*Theta*Sigma);
assert(norm(g2-g) < 1e-2)

f = @(x) -trace(Sigma*reshape(x,[],p)'*Sxx*reshape(x,[],p)*Sigma*U);
g = numdiff(f, vec(Theta));
g2 = vec(-2*Sxx*Theta*Sigma*U*Sigma);
assert(norm(g2-g) < 1e-2);

% test Theta / Lambda second differentials together
f = @(x) obj_vec(x, S, p);
[g H] = numdiff(f, [vec(Lambda); vec(Theta)]);
d2f = trace(Sigma*U*Sigma*U) + 2*trace(Sigma*V'*Sxx*V) - ...
      4*trace(Sigma*U*Sigma*Theta'*Sxx*V) + ...
      2*trace(Sigma*U*Sigma*Theta'*Sxx*Theta*Sigma*U);
d = [vec(U); vec(V)];
assert(abs(d'*H*d - d2f) < 1e-2);

% test the newton_obj function
obj = newton_obj(Lambda, Theta, U, V, S, Sigma);
assert(abs(g'*d + 0.5*d'*H*d - obj) < 1e-2);

disp('PASSED');