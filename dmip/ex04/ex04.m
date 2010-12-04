%% cleanup
close all;
clear all;
clc;

%% init
% either image (+ cropping)
I = imread('mr12.tif');
I = I(40:248,25:198);

% or checkerboard
%I = checkerboard(10) > 0.5;

%% normalize and plot
I = mat2gray(I);
subplot(2,3,1), colormap gray, imagesc(I), title('original');

%% create ramp
ramp = ones(size(I));
for i = 1:size(ramp, 1)
    ramp(size(ramp, 1)+1 - i, :) = i / size(ramp, 1); 
end
ramp = 0.99 * ramp + ones(size(ramp)) * 0.001;
subplot(2,3,2), imagesc(ramp), title('linear ramp');

%% corrupt image...
corrupted = I .* ramp;
subplot(2,3,3), imagesc(corrupted), title('corrupted');

%% compute mean-values
tic
neighborhood = [50 50];
mu_overall = mean(corrupted(:));
mu_per_pixel = imfilter(corrupted, fspecial('average', neighborhood), 'replicate');

%% compute HUM-optimized image
corrected = corrupted .* ((ones(size(mu_per_pixel)) .* mu_overall) ./ mu_per_pixel);
toc
subplot(2,3,4), imagesc(corrected), title('HUM corrected');

%% further plots
subplot(2,3,5), imagesc(I - corrected), title('original - corrected');
subplot(2,3,6), imagesc(corrupted - corrected), title('corrupted - corrected');
