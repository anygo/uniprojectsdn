% automatic thresholding algorithm (see intropr preprocessing, slide 44)
close all;
clear all;
clc;

% create foreground/background image
im = randn(256, 256);
im(16:128, 128:240) = im(16:128, 128:240) + 5;
im = mat2gray(im);
imagesc(im);
colormap gray;

% 1. initialize theta to the average gray value
theta_new = mean(im(:));
theta_old = -1;

% loop
while abs(theta_old - theta_new) > 1e-4
    % 2. partition the data around theta and compute the mean value for each
    % partition
    partition_1 = (im(im <= theta_new));
    mu_1 = mean(partition_1(:));
    partition_2 = (im(im > theta_new));
    mu_2 = mean(partition_2(:));

    % 3. theta = mean of the two means
    theta = 0.5*(mu_1 + mu_2);
    
    theta_old = theta_new;
    theta_new = theta;
end

imagesc(im <= theta);