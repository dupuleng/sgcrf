function cv = cvpartition_grouped(m, k, g)

cv0 = cvpartition(ceil(m/g), 'kfold', k);

cv.N = 20;
cv.NumTestSets = k;

for i=1:k,
  tr = kron(cv0.training(i), ones(g,1));
  te = kron(cv0.test(i), ones(g,1));
  cv.training(:,i) = logical(tr(1:m));
  cv.test(:,i) = logical(te(1:m));
  cv.TrainSize(i) = nnz(cv.training(:,i));
  cv.TestSize(i) = nnz(cv.test(:,i));
end


  
