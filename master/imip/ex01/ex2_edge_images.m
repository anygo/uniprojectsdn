% cleanup
close all;
clear all;
clc;

% read image
I = dicomread('mr14.dcm');
I = mat2gray(I);
subplot(2,2,1);
imagesc(I);
title('$I$','interpreter','latex');
colormap gray;

% apply sobel
Ix = imfilter(I, fspecial('Sobel')');
Iy = imfilter(I, fspecial('Sobel'));

subplot(2,2,3);
imagesc(Ix);
title('$I_x = \partial{}I/\partial{}x$','interpreter','latex');

subplot(2,2,4);
imagesc(Iy);
title('$I_y = \partial{}I/\partial{}y$','interpreter','latex');

% gradient magnitude image
Igm = sqrt(Ix.^2 + Iy.^2);

subplot(2,2,2);
imagesc(Igm);
title('$\sqrt{{I_x}^2 + {I_y}^2}$','interpreter','latex')