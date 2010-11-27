%% aufraeumen   
clear all;
%close all;
clc;
pausetime = 0.5;
figure(1);
colormap gray;
    
%% bild laden    
img = rgb2gray(imread('bild_sb/b (9).jpg'));
imagesc(img);
pause(pausetime);
img = imfilter(img, fspecial('gaussian', [25 25]));
imagesc(img);
pause(pausetime);
laplacian = imfilter(img, fspecial('laplacian'), 'replicate');
imagesc(laplacian);
pause(pausetime);

sigma = 2;
hsize = [5 5];
img2 = imfilter(img, fspecial('log', hsize, sigma));
imagesc(img2);
pause(pausetime);

%% thresholding
imgbw = im2bw(img2, graythresh(img2)); 
subplot(2,1,1), imagesc(img + uint8(~imgbw)*100), subplot(2,1,2), imagesc(imgbw);
drawnow;
pause(pausetime);

%% wildes morphing
imgbw = imclose(imgbw, strel('disk', 12));
subplot(2,1,1), imagesc(img + uint8(~imgbw)*100), subplot(2,1,2), imagesc(imgbw);
drawnow;
pause(pausetime);

imgbw = imopen(imgbw, strel('disk', 50));
subplot(2,1,1), imagesc(img + uint8(~imgbw)*100), subplot(2,1,2), imagesc(imgbw);
drawnow;
pause(pausetime);

imgbw = imclose(imgbw, strel('disk', 50, 0));
subplot(2,1,1), imagesc(img + uint8(~imgbw)*100), subplot(2,1,2), imagesc(imgbw);
drawnow;
pause(pausetime);


%% kleine zellen finden
segmented = img .* uint8(imgbw);
figure(2);
imagesc(histeq(segmented));
title('krypten ausgeschnitten');
colormap gray;


