% author: Dominik Neumann and nobody else (some people hate me for that)

% entropy based binarization (see intropr preprocessing, slide 38ff)
%close all;
clear all;
clc;

% create foreground/background image
im = [randn(256, 256) randn(256, 100)+5];


%im = imread('rabbit.jpg');
im = mat2gray(im);

subplot(2,3,1);
imagesc(im);
subplot(2,3,2);
imhist(im);

colormap gray;

% create histogram
[counts,x] = imhist(im, 128);
counts = counts/(sum(counts));

theta_opt = 0;
cur_max = -Inf;

h1s = zeros(1,size(x,1));
h2s = zeros(1,size(x,1));

for pos = 1:size(x,1)
    theta = x(pos);
    D_theta = sum(counts(x <= theta));
    
    H_1 = -sum((counts(x <= theta & counts > 0)/D_theta)    .* log(counts(x <= theta & counts > 0)/D_theta));
    H_2 = -sum((counts(x > theta & counts > 0)/(1-D_theta)) .* log(counts(x > theta & counts > 0)/(1-D_theta)));
    
    h1s(1,pos) = H_1;
    h2s(1,pos) = H_2;
    
    subplot(2,3,4);
    plot(x,h1s);
    hold on;
    plot(x,h2s,'r');
    plot(x,h1s + h2s,'--black');
    hold off;
    subplot(2,3,5);
    imhist(im(im <= theta));
    subplot(2,3,6);
    imhist(im(im > theta));
    
    subplot(2,3,3);
    imagesc(im <= theta);
    drawnow;
    
    if H_1 + H_2 > cur_max
        theta_opt = theta;
        cur_max = H_1 + H_2;
    end
end

subplot(2,3,3);
imagesc(im <= theta_opt);
subplot(2,3,5);
imhist(im(im <= theta_opt));
subplot(2,3,6);
imhist(im(im > theta_opt));
