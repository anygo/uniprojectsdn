%% cleanup
clear all;
close all;
clc;

%% ex13
I = double(imread('object.png'));
%I = imerode(I, strel('disk', 3, 0));
%I = imdilate(I, strel('disk', 3, 0));
I = imfilter(I, fspecial('gaussian',5));
sobel_x = [-1 0 1; -2 0 2; -1 0 1];
sobel_y = [-1 -2 -1; 0 0 0; 1 2 1];

dx = imfilter(I, sobel_x);
dy = imfilter(I, sobel_y);
magnitude = sqrt(dx.^2+dy.^2);
imagesc(magnitude);
colormap gray;