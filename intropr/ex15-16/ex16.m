%% cleanup
clear all;
close all;
clc;

%% Initialisierung
%I = imread('Lena.png');
I = imread('LenaLowContrast.png');
bins = 2^8;
figure(1);
colormap gray;
subplot(2,2,1), imagesc(mat2gray(I)), axis off, title('orig');

%% cdf berechnen
[cdf h] = compute_cdf(I, bins);
subplot(2,2,2), stairs(1:bins, cdf), axis([0 bins 0 1]), title('cdf');
subplot(2,2,3), stairs(1:bins, h), title('hist orig');

%% neues histogramm erstellen
steps = zeros(bins,1);
for i = 1:bins
    upper_bound = i * (1/bins);
    steps(i) = size(find(cdf < upper_bound),1) + 1;
end

%% Bild equalizen
I_eq = uint8(zeros(size(I)));
cur = -1;
for i = 1:bins
    if steps(i) > cur
        I_eq(I > cur & I <= steps(i)) = i;
        cur = steps(i);
    end
end
subplot(2,2,4), imagesc(I_eq), axis off, title('equalized');