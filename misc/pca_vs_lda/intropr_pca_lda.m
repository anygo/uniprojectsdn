%% PCA
clear all
close all
clc

load('pca_data.mat')
%load('stripedData1.mat')
%load('stripedData2.mat')

% Compute Eigenvector Matrix straightaway via princomp command
help=princomp(patterns');


% Covariance matrix Q = sum_i(sum_j((fi-fj) * (fi-fj)'))
Q=zeros(size(patterns,1));
for i=1:size(patterns,2)
    for j=1:size(patterns,2)
        Q=Q+(patterns(:,i)-patterns(:,j))*(patterns(:,i)-patterns(:,j))';
    end
end
%Q=Q/size(patterns,2)^2;

[V,D]=eigs(Q);

% Create transformation matrix.
Phi=V';
%Phi(2,:)=0;  % If you want to ignore the y dimension

for k=1:size(patterns,2)
    patterns_PCA(:,k)=Phi*patterns(:,k);
end

figure(1)
subplot(1,2,1)
scatter(patterns(1,classes==0),patterns(2,classes==0),'+','b');
hold on
scatter(patterns(1,classes==1),patterns(2,classes==1),'o','b');
hold on
Phihelp=Phi/2; %Scale Phi for visualisation
plot([0,Phihelp(1,1)],[0,Phihelp(1,2)])
plot([0,Phihelp(2,1)],[0,Phihelp(2,2)])
axis image
subplot(1,2,2)
scatter(patterns_PCA(1,classes==0),patterns_PCA(2,classes==0),'+','r')
hold on
scatter(patterns_PCA(1,classes==1),patterns_PCA(2,classes==1),'o','r')
axis image
title('PCA')

%% LDA

% Compute Eigenvector Matrix straightaway via princomp command
[help]=princomp(patterns');


% Create class dependent means
MeanClass0=mean(patterns(:,classes==0)')';
MeanClass1=mean(patterns(:,classes==1)')';
Mean=mean(patterns')';

sum0=zeros(2);
sum1=zeros(2);
for i=1:size(classes,2)
    if classes(i)==0
        sum0=sum0+(patterns(:,i)-MeanClass0)*(patterns(:,i)-MeanClass0)';
    else
        sum1=sum1+(patterns(:,i)-MeanClass1)*(patterns(:,i)-MeanClass1)';
    end
end
SW=sum0+sum1;
SB=(size(classes,2)-nnz(classes))*((MeanClass0-Mean)*(MeanClass0-Mean)');
SB=SB+nnz(classes)*((MeanClass1-Mean)*(MeanClass1-Mean)');

[V, D]=eigs(SW\SB);

% Create transformation matrix.
Phi=V';
%Phi(2,:)=0;  % If you want to ignore the y dimension
Phi(1,:)=Phi(1,:)*-1; % To enforce same "rotation" direction than PCA does (visualisation)

for k=1:size(patterns,2)
    patterns_LDA(:,k)=Phi*patterns(:,k);
end

figure(2)
subplot(1,2,1)
scatter(patterns(1,classes==0),patterns(2,classes==0),'+','b');
hold on
scatter(patterns(1,classes==1),patterns(2,classes==1),'o','b');
hold on
Phi=Phi/2; %Scale Phi for visualisation
plot([0,Phi(1,1)],[0,Phi(1,2)])
plot([0,Phi(2,1)],[0,Phi(2,2)])
axis image
subplot(1,2,2)
scatter(patterns_LDA(1,classes==0),patterns_LDA(2,classes==0),'+','r')
hold on
scatter(patterns_LDA(1,classes==1),patterns_LDA(2,classes==1),'o','r')
axis image
title('LDA')