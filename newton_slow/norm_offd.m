function x = norm_offd(A)
x = norm(A - diag(diag(A)),1);