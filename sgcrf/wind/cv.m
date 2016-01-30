function [errs par_mat] = cv(X, Y, features, pred, err, params, split) 
[m p] = size(Y);
          
par0 = params;
par_cell = struct2cell(params);
par_grid = cell(1,length(par_cell));
[par_grid{:}] = ndgrid(par_cell{:});
clear par_mat;
for i=1:length(par_cell)
  par_mat(:,i) = par_grid{i}(:);
end
f = fieldnames(params);

[iy jy] = find(isnan(Y));

errs = zeros(size(par_mat,1),2);
err_fold = zeros(split.NumTestSets,2);
for i = 1:size(par_mat,1),
  for j=1:length(f), par0.(f{j}) = par_mat(i,j); end
  
  Phi = features(X, par0, struct);
  [ip jp] = find(isnan(Phi));
  
  for j=1:split.NumTestSets
    t = setdiff(find(split.training(:,j)), union(iy, ip));
    s = setdiff(find(split.test(:,j)), union(iy, ip));

    [Pt Ps] = pred(Phi(t,:), Y(t,:), Phi(s,:), par0);
    
    % rmse, weighted by example
    err_fold(j,:) = [err(Y(t,:), Pt) err(Y(s,:), Ps)];
  end
    
  errs(i,:) = mean(err_fold);
  fprintf('%2d %f %f\n', i, errs(i,1), errs(i,2)); 
end




