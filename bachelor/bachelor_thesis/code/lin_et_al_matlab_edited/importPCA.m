function [PCA_mean, PCA_base] = importPCA(datfile)

% [PCA_mean, PCA_base] = importPCA(datfile)
% 
% Import PCA model of radiometric response function.
%
% Jinwei Gu. Last modified: 2003-10-8

if nargin == 0
    datfile = 'meanCurve.dat';
end

fp = fopen(datfile,'r');
M = fscanf(fp,'%d',1); % bin number, default is 1024
N = 25;
t = fscanf(fp, '%f ', M);
PCA_mean = fscanf(fp, '%f ', M);
PCA_base = zeros(N,M);

for i=1:N
    t = fscanf(fp, '%f ', M);
    PCA_base(i,:) = t';
end
fclose(fp);