% Mutual incoherence for CRF chain graph
bs = [0.1 1 5 10];
as = [logspace(-4, log10(0.3), 50)];
ax = [];
h = [];
 
ns = [3 11 12 15];
X = zeros(length(bs), length(as));
  
for i=1:length(bs)
  b = bs(i);
  for j=1:length(as)
    a = as(j);
    Lambda = [1 a 0;
              a 1 a;
              0 a 1];
    Sigma = inv(Lambda);
    Theta = a*b*eye(3);
    Sxx = eye(3);
    
    s = find([vec(Lambda); vec(Theta)] ~= 0);
    
    H_LL = kron(Sigma, Sigma + 2*Sigma*Theta'*Sxx*Theta*Sigma);
    H_LT = kron(-2*Sigma, Sigma*Theta'*Sxx);
    H_TT = kron(2*Sigma, Sxx);
    H = [H_LL H_LT; H_LT' H_TT];

    
    X(i,j) = max(sum(abs(H(ns,s)*inv(H(s,s))),2));
  end
end
  
figure;
plot(as, X, 'LineWidth', 1);
axis([0 0.3 0 1]);
legend('\beta = 0.1', '\beta = 1', '\beta = 5', '\beta = 10', ...
       'Location', 'SouthEast')
prepare_figure('chain_incoherence.pdf', [4 3], '\rho', 'Mutual incoherence')


