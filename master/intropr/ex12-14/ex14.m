%% cleanup
clear all;
close all;
clc;

%% ex14
I = double(imread('object.png'));
angle = 270;

%% part a
Ia = I;
figure(1), colormap gray;
subplot(2,2,1), imagesc(Ia), axis off;
for i = 1:angle
    Ia = imrotate(Ia, 1, 'bilinear', 'crop');
    imagesc(Ia); axis off; drawnow;
end

%% part b
Ib = I;
Ib = imrotate(Ib, angle, 'bilinear', 'crop');
subplot(2,2,2), imagesc(Ib), axis off;

%% part c
sobel_x = [-1 0 1; -2 0 2; -1 0 1];
sobel_y = [-1 -2 -1; 0 0 0; 1 2 1];

% a
dxa = imfilter(Ia, sobel_x);
dya = imfilter(Ia, sobel_y);
magnitudea = sqrt(dxa.^2+dya.^2);
subplot(2,2,3), imagesc(magnitudea), axis off;

% b
dxb = imfilter(Ib, sobel_x);
dyb = imfilter(Ib, sobel_y);
magnitudeb = sqrt(dxb.^2+dyb.^2);
subplot(2,2,4), imagesc(magnitudeb), axis off;

pause;

% a2
lapa = imfilter(Ia, fspecial('laplacian'));
subplot(2,2,3), imagesc(lapa), axis off;

% b2
lapb = imfilter(Ib, fspecial('laplacian'));
subplot(2,2,4), imagesc(lapb), axis off;