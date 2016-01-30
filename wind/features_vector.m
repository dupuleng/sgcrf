function [Phi, feature_params] = features_vector(X,params,feature_params)
[m n] = size(X);

s = find(mod([1:n],3) == 0); % magnitude
u = find(mod([1:n],3) == 1); 
v = find(mod([1:n],3) == 2);

Xs = X(:,s);
Xu = X(:,u);
Xv = X(:,v);

if isfield(feature_params, 'mu')
  mu = feature_params.mu;
else
  if isfield(params, 'rbf_d')
    d = params.rbf_d;
    % min_u = min(Xu(:));
    % max_u = max(Xu(:));
    % min_v = min(Xv(:));
    % max_v = max(Xv(:));
    
    % wu = (max_u - min_u)/d/2;
    % wv = (max_v - min_v)/d/2;
    % [mu_u mu_v] = meshgrid(linspace(min_u+wu, max_u-wu, d), ...
    %                        linspace(min_v+wv, max_v-wv, d));  
    % mu = [mu_u(:) mu_v(:)];
    
    mu = zeros(n/3*d^2, 2);
    for i=1:n/3
      min_u = min(Xu(:,i));
      max_u = max(Xu(:,i));
      min_v = min(Xv(:,i));
      max_v = max(Xv(:,i));
      
      wu = (max_u - min_u)/d/2;
      wv = (max_v - min_v)/d/2;
      [mu_u mu_v] = meshgrid(linspace(min_u+wu, max_u-wu, d), ...
                             linspace(min_v+wv, max_v-wv, d));  
      
      mu((i-1)*d^2+1:i*d^2,:) = [mu_u(:) mu_v(:)];
    end
  else
    rand('state', 1);
    k = params.rbf_mus;
    mu = zeros(n/3*k, 2);
    for i = 1:n/3
      mu((i-1)*k+1:i*k, :) = centers([Xu(:, i) Xv(:, i)], params.rbf_mus);
    end
  end
  
  feature_params = struct('mu', mu);
end

width = params.bandwidth;

k = size(mu,1)/n*3;
X_rbf = zeros(m,n/3*k);
for i = 1:n/3
  D = sqdist([Xu(:, i) Xv(:, i)]', mu((i-1)*k+1:i*k, :)');
  X_rbf(:, (i-1)*k+1:i*k) = exp(-0.5/width^2 * D);
end

Phi = [X_rbf ones(m,1)];


