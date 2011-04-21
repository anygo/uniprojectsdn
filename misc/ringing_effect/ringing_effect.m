% cleanup
close all;
clear all;
clc;

% what is the ringing effect?
% gauss filter vs. mean filter

% create filters
gauss_kernel = fspecial('gaussian', [1 64], 2);
mean_kernel = ones(1, 64)/64;

% zeropad it
gauss_kernel = padarray(gauss_kernel, [0 32]);
mean_kernel = padarray(mean_kernel, [0 32]);

subplot(3,2,1);
plot(gauss_kernel);
title('Gauss-Filter');
ylabel('spatial domain');
subplot(3,2,2);
plot(mean_kernel);
title('Mean-Filter');

% fourier-transform
gauss_kernel_fourier = fftshift(fft(gauss_kernel));
mean_kernel_fourier = fftshift(fft(mean_kernel));

subplot(3,2,3);
plot(abs(gauss_kernel_fourier));
title('-> no ringing');
ylabel('frequency domain');
subplot(3,2,4);
plot(abs(mean_kernel_fourier));
title('-> ringing');

% create test image
im = zeros(128,128);
im(20:44,20:44) = 1;
im(:,5) = 1;
im(5,:) = 1;
im(8,:) = 1;
im(100,:) = 1;
im(1:3:end, 2:3:end) = 1;

% 2D-filter
im_gauss = imfilter(im, fspecial('gaussian', [15 15], 3));
im_mean = imfilter(im, fspecial('average', [15 15]));

subplot(3,2,5);
imagesc(im_gauss);
axis image;
subplot(3,2,6);
imagesc(im_mean);
axis image;
colormap gray;

% examples are not thaaat good ;)
% I don't care!