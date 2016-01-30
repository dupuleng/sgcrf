function fx = obj_vec(x, S, p)
Lambda = reshape(x(1:p*p), [], p);
Theta = reshape(x(p*p+1:end), [], p);

fx = -log(det(Lambda)) + trace(Lambda*S(1:p,1:p)) + ...
     2*trace(Theta*S(1:p, p+1:end)) + ...
     trace(Theta/Lambda*Theta'*S(p+1:end,p+1:end));
                                                 
