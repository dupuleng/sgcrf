function Phi = features_forecast(X)

% Put RBFs over the temperatures at noon every day.
%X_prev_temp = X.prev_temp(:, 168-12+1:168:end);
X_temp = X.temp(:, 12:6:end);
d = 3;
n = size(X_temp, 2);
X_temp_rbf = zeros(X.m, n*d);
for i=1:n
  j = (i-1)*d+1;
  k = j+(d-1);
  X_temp_rbf(:, j:k) = phi_rbf(X_temp(:,i), d);
end

Phi = [ones(X.m,1) ...
       X.time ...
       X.prev_week ...
       phi_rbf_cyc(X.week, 3) ... 
       phi_rbf_cyc(X.year, 10) ...
       X_temp_rbf ...
      ];


       
