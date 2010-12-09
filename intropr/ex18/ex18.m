%% cleanup
clear all;
%close all;
clc;

%% ex 18 - moments
% bild laden und "binarizen"
I = imread('momented.png');
I = im2bw(I, 0.5);
subplot(1,3,1), imagesc(I), colormap gray;

% image moments berechnen für centroid
computeCentralMoment = 0;
m_00 = compute_moments(I, 0, 0, computeCentralMoment);
m_10 = compute_moments(I, 1, 0, computeCentralMoment);
m_01 = compute_moments(I, 0, 1, computeCentralMoment);

centroid_x = m_10/m_00;
centroid_y = m_01/m_00;

% in die mitte verschieben
[y x] = find(I == 1);
I_translated = zeros(size(I));
x = floor(x - centroid_x) + 150;
y = floor(y - centroid_y) + 150;
for i = 1:size(x,1);
    I_translated(y(i), x(i)) = 1/m_00;
end
subplot(1,3,2), imagesc(I_translated);

% covarianzen berechnen zur rotation
computeCentralMoment = 1;
m_20 = compute_moments(I_translated, 2, 0, computeCentralMoment) / size(find(I_translated > 0), 1);
m_02 = compute_moments(I_translated, 0, 2, computeCentralMoment) / size(find(I_translated > 0), 1);
m_11 = compute_moments(I_translated, 1, 1, computeCentralMoment) / size(find(I_translated > 0), 1);

rotation = 0.5 * atan(2*m_11 / (m_20 - m_02)) * 180 / pi;

I_rotated = imrotate(I_translated, rotation, 'crop');
subplot(1,3,3), imagesc(I_rotated);