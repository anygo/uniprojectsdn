%% cleanup
close all
clear all
clc

%% params
dim = 22;
n_features = 250;

%% feature matrix
f = rand(dim,n_features);
f2 = f;

for i = 1:dim-2
    tic;
    i;
    f = pca_dimensionality_reduction(f,1);
    toc;
end;

f2 = pca_dimensionality_reduction(f2,dim-2);


%% plot
scatter(f2(1,:),f2(2,:),'+','red');
hold on;
scatter(f(1,:),f(2,:),'x','green');
hold off;