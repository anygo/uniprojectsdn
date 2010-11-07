close all;
clear all;

I = phantom(128);
A = fft2(I);
B = (abs(A));
C = log(1+abs(A));

subplot(2,2,1);
imagesc(I);
subplot(2,2,2);
imagesc(fftshift(abs(A)));
subplot(2,2,3);
imagesc(C);
subplot(2,2,4);
imagesc(fftshift(C));
colormap gray;

% fftshift -> put center of beam into center