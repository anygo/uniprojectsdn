%% cleanup
close all;
clear all;
clc;

% samples
X = [1 2; 1.75 2.5; 2 1; 2 3; 2.5 2; 2.5 2.5; 3 1]';

% a) sample mean and sample variance of X
mu_all = mean(X,2)
var_all = var(X')'

% b) LOOCV
loo_mus = zeros(size(X));
loo_vars = zeros(size(X));
for left_out = 1:size(X, 2)
    considered = X(:,(1:end)~=left_out);
    loo_mus(:,left_out) = mean(considered,2);
    loo_vars(:,left_out) = var(considered')';
end
var_of_mus = var(loo_mus')'
var_of_vars = var(loo_vars')'

% c) 2-Fold CV
% just split into two parts???
set_1 = X(:,1:round(size(X,2)/2));
set_2 = X(:,round(size(X,2)/2)+1:end);

mu_set1 = mean(set_1,2);
mu_set2 = mean(set_2,2);
var_set1 = var(set_1')';
var_set2 = var(set_2')';

var_of_mus_2fold = mean([mu_set1 mu_set2],2)
var_of_vars_2fold = var([mu_set1 mu_set2]')'

% d) Bias Variance Trade-Off?