%% aufraeumen   
clear all;
%close all;
clc;
    
%% bild laden    
img = rgb2gray(imread('bild_sb/b (9).jpg'));
imagesc(img);
%pause(0.5);
img = imfilter(img, fspecial('gaussian', [25 25]));
imagesc(img);
%pause(0.5);
laplacian = imfilter(img, fspecial('laplacian'), 'replicate');
imagesc(laplacian);
%pause(0.5);

sigma = 2;
hsize = [5 5];
img2 = imfilter(img, fspecial('log', hsize, sigma));
imagesc(img2);
%pause(0.5);

%% thresholding
imgbw = im2bw(img2, graythresh(img2)); 

%% dilate
%imgbw = imdilate(imgbw, strel('disk', 5, 0));
imagesc(imgbw);
drawnow;

imgbw = imclose(imgbw, strel('disk', 12));
imagesc(imgbw);
drawnow;

imgbw = imopen(imgbw, strel('disk', 45));
imagesc(img.*uint8(~imgbw))
drawnow;

colormap gray;

