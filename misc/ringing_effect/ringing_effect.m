% cleanup
close all;
clear all;
clc;

% what is the ringing effect?
% gauss filter vs. mean filter

% create filters
gauss_kernel = fspecial('gaussian', [1 25], 2);
mean_kernel = ones(1, 25)/25;

% zeropad it
gauss_kernel = padarray(gauss_kernel, [0 25]);
mean_kernel = padarray(mean_kernel, [0 25]);

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

% test image
im = zeros(64,64);
im(20:44,20:44) = 1;
im(:,5) = 1;
im(5,:) = 1;

% filter
im_gauss = imfilter(im, fspecial('gaussian', [3 3], 0.6));
im_mean = imfilter(im, fspecial('average', [3 3]));

subplot(3,2,5);
imagesc(im_gauss);
axis image;
subplot(3,2,6);
imagesc(im_mean);
axis image;
colormap gray;

% examples are not thaaat good ;)
% I don't care!