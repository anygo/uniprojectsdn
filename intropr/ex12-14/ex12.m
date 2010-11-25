%% cleanup
clear all;
close all;
clc;

%% ex12
I = imread('moon.jpg');
mask = [1 1 1; 1 -8 1; 1 1 1];
filtered = imfilter(I,mask,'replicate');
enhanced = I - filtered;
subplot(1,3,1); imagesc(I); axis off;
subplot(1,3,2); imagesc(filtered); axis off;
subplot(1,3,3); imagesc(enhanced); axis off;
colormap gray;