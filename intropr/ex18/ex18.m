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
x = floor(x - centroid_x) + round(size(I, 2)/2);
y = floor(y - centroid_y) + round(size(I, 1)/2);
for i = 1:size(x,1);
    I_translated(y(i), x(i)) = 1/m_00;
end
subplot(1,3,2), imagesc(I_translated);

% covarianzen berechnen zur rotation
computeCentralMoment = 1;
mu_20 = compute_moments(I_translated, 2, 0, computeCentralMoment) / size(find(I_translated > 0), 1);
mu_02 = compute_moments(I_translated, 0, 2, computeCentralMoment) / size(find(I_translated > 0), 1);
mu_11 = compute_moments(I_translated, 1, 1, computeCentralMoment) / size(find(I_translated > 0), 1);

% mit wikipedia:
%rotation = 0.5 * atan(2*mu_11 / (mu_20 - mu_02)) * 180 / pi;

% andere variante.. selbst erprobt :-)
cov_mat = [mu_20 mu_11; mu_11 mu_02];
[U S V] = svd(cov_mat);
vec = U(:,1);
hypothenuse = norm(vec, 2);
gegenkathete = [1;0]'*vec;
winkel = -asin(gegenkathete / hypothenuse) * 180 / pi;
hold on;
plot([150 150+300*S(1,1)*U(1,1)], [150 150+300*S(1,1)*U(2,1)]);
plot([150 150+300*S(2,2)*U(1,2)], [150 150+300*S(2,2)*U(2,2)]);
hold off;

I_rotated = imrotate(I_translated, winkel, 'crop');
subplot(1,3,3), imagesc(I_rotated);