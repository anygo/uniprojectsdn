%% cleanup
clear all;
close all;
clc;

%% load image
%im = phantom(256);
%im = zeros(64, 64);
%im(20:40, 30:50) = 1;
im = imread('Lenna.png');
im = im2bw(im);
im = imresize(im, 0.5);
subplot(2, 2, 1);
imagesc(im);
title('input image');
colormap gray;
drawnow;

%% edge detection
Jx = imfilter(im, fspecial('Sobel')');
Jy = imfilter(im, fspecial('Sobel'));

%% structure tensor
n = 16;
im_outA = zeros(size(im));
im_outB = zeros(size(im));

h = waitbar(0, 'please wait');
for y = 1+ceil(n/2):size(im, 1)-n/2
    for x = 1+ceil(n/2):size(im, 2)-n/2
        tmp_mat = zeros(2, 2);
        for u = -n/2:n/2
            for v = -n/2:n/2 
                tmp_mat(1, 1) = tmp_mat(1, 1) + Jx(y+v, x+u)^2;
                tmp_mat(1, 2) = tmp_mat(1, 2) + Jx(y+v, x+u) * Jy(y+v, x+u);
                tmp_mat(2, 1) = tmp_mat(2, 1) + Jx(y+v, x+u) * Jy(y+v, x+u);
                tmp_mat(2, 2) = tmp_mat(2, 2) + Jy(y+v, x+u)^2;
            end
        end
        d = eigs(tmp_mat);
        im_outA(y, x, 1) = d(1);
        im_outB(y, x, 1) = d(2);
    end 
    waitbar(y/(size(im, 1) - n));
end
close(h);

subplot(2, 2, 3);
imagesc(im_outA);
title('largest eigenvalue');
subplot(2, 2, 4);
imagesc(im_outB);
title('smallest eigenvalue');
drawnow;

%% thresholding
tc = 15;
im_out = (im_outA > tc) & (im_outB > tc);
subplot(2, 2, 2);
imagesc(im_out(:,:,1));
title('output (corner) image');