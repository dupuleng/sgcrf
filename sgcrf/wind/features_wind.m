function [Phi, fparams] = features_ar(X,params,fparams)
[m n] = size(X);

i = params.num_ar_features;
Xwf = X(:, i+1:end);
[Phi_wf fparams] = features_vector(Xwf,params,fparams);
Phi = [X(:,1:i) Phi_wf];

