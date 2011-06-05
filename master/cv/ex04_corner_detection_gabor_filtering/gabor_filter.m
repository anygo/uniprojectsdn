% cleanup
clear all;
close all;
clc;

% load image
%im = phantom(256);
%im = zeros(64, 64);
%im(20:40, 30:50) = 1;
im = imread('Lenna.png');
im = im2bw(im);
im = imresize(im, 0.5);
subplot(2, 2, 1);
imagesc(im);
title('input image');
colormap gray;
drawnow;

for theta = pi/3:pi/8:3*pi/2
    for lambda = 0.4:0.2:4
        % compute the gabor filter mask
        %lambda = 1.4;
        %theta = 7;
        sigma = 8;
        kernel_size = 25;
        kernel_size_half = floor(kernel_size/2);
        kernel = ones(kernel_size);
        for y = -kernel_size_half:kernel_size_half
            for x = -kernel_size_half:kernel_size_half
                x_prime = x*cos(theta) + y*sin(theta);
                y_prime = -x*sin(theta) + y*cos(theta);
                kernel(y+kernel_size_half+1, x+kernel_size_half+1) = ...
                    (1 / sqrt(2*pi*sigma^2)) * ...
                    exp(-(x_prime^2 + y_prime^2)/sigma^2) * ...
                    cos(x_prime / lambda);
            end
        end
        subplot(2, 2, [3 4]);
        [X Y] = meshgrid(-kernel_size_half:kernel_size_half, -kernel_size_half:kernel_size_half);
        surf(X, Y, kernel);
        colormap gray;

        % convolve the image with the gabor kernel
        im_filtered = imfilter(im, kernel);
        subplot(2, 2, 2);
        imagesc(im_filtered);
        drawnow;
        %pause(0.01);
    end
end