%% cleanup
close all
clear all
clc

%% params
dim = 100;
n_features = 25000;

%% feature matrix
f = rand(dim,n_features);
f2 = f;

for i = 1:dim-3
    tic;
    i;
    f = pca_dimensionality_reduction(f,1);
    toc;
end;

f2 = pca_dimensionality_reduction(f2,dim-3);


%% plot
scatter3(f2(1,:),f2(2,:),f2(3,:),'+','red');
hold on;
scatter3(f(1,:),f(2,:),f(3,:),'x','green');
hold off;