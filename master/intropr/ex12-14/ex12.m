%% cleanup
clear all;
close all;
clc;

%% ex12
I = imread('moon.jpg');
I = imfilter(I, fspecial('gaussian'));

mask = [1 1 1; 1 -8 1; 1 1 1];
filtered = uint8(imfilter(double(I),mask,'replicate'));
enhanced = I - filtered;
subplot(2,3,1); imagesc(I); axis off, title('orig');
subplot(2,3,2); imagesc(filtered); axis off, title('laplacian');
subplot(2,3,3); imagesc(enhanced); axis off, title('enhanced');
colormap gray;


% oder mit sobel:
dx = imfilter(I,fspecial('sobel'),'replicate');
dy = imfilter(I,fspecial('sobel')','replicate');
filtered2 = sqrt(double(dx).^2+double(dy).^2);

enhanced = I - uint8(filtered2);
subplot(2,3,4); imagesc(I); axis off, title('orig');
subplot(2,3,5); imagesc(filtered2); axis off, title('sobel');
subplot(2,3,6); imagesc(enhanced); axis off, title('enhanced');
colormap gray;