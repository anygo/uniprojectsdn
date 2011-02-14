% exercise 8
close all;
clear all; 
clc;

n_bits = 2;
max_iter = 15;


%% read image
I = imread('image.jpg');
I = imresize(I,0.01);

%% do stuff
I3d = together(I, 3*n_bits, max_iter);
Isep = separately(I, n_bits, max_iter);


%% plot
subplot(1,3,1)
imagesc(I);
axis off;
title('orig');
subplot(1,3,2)
imagesc(I3d);
axis off;
title('3d');
subplot(1,3,3)
imagesc(Isep);
axis off;
title('separat');