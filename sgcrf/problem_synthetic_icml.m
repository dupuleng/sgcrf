function [X Y t s] = problem_synthetic_icml()

m = 2500;
n = 4000;
p = 1000;

randn('seed', 1);
rand('seed', 1);

[Lambda0 Theta0] = gen_params3(n, p);
[X Y] = gen_data(Lambda0, Theta0, m*2);

t = 1:m;
s = m+1:2*m;
