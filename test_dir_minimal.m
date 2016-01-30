function test_dir_minimal(Lambda, Theta, U, V, S, lambda, Sigma)
[n p] = size(Theta);

% check U
f = @(x) newton_obj(Lambda, Theta, make_sym(reshape(x, [], p)), V, S, Sigma);
gradU = reshape(numdiff(f, vec(U)), [], p);
a = triu(U~=0|Lambda~=0,1);
nz = (abs(Lambda+U)>1e-4)&a;
z =  (abs(Lambda+U)<1e-4)&a;
assert(norm(diag(gradU)) < 1e-2);
assert(all(abs(gradU(z)) < lambda));
assert(norm(abs(gradU(nz)) - lambda) < 1e-2);
assert(all(sign(gradU(nz)) == -sign(Lambda(nz) + U(nz))));

% check V
f = @(x) newton_obj(Lambda, Theta, U, reshape(x, [], p), S, Sigma);
gradV = reshape(numdiff(f, vec(V)), [], p);
a = V~=0|Theta~=0;
nz = (Theta+V)&a;
z = ~(Theta+V)&a;
assert(all(abs(gradV(z)) < lambda));
assert(norm(abs(gradV(nz)) - lambda) < 1e-2);
assert(all(sign(gradV(nz)) == -sign(Theta(nz) + V(nz))));

