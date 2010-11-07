%% cleanup
close all;
clear all;

%% load and normalize image to [0;1]
I = double(imread('undistorted.jpg'));
I = I - ones(size(I))*min(min(I));
I = I/max(max(I));

%% cut image (-> square-matrix)
Icut = I(1:min(size(I)),1:min(size(I)));

%% plot cutted image
subplot(2,2,1);
imagesc(Icut);
grid on;
colormap gray;
title('cutted image');

%% create grid for sampling
[X,Y] = meshgrid(1:1:size(Icut,1), 1:1:size(Icut,2));

%% create distortion field (R)
a = 5;
b = 15;
d = 10;
R = d*sqrt(a*((X-size(X,1)/2)/size(X,1)).^2 + b*((Y-size(X,1)/2)/size(X,1)).^2); 

%% shift distortion field
omega = max(max(R));
R = R - ones(size(R))*omega/2; 

%% plot distortion field
subplot(2,2,2);
imagesc(R);
colormap gray;
colorbar;
axis equal; axis off;
title('distortion field');

%% add distortion field to grid
XI = X+R;
YI = Y+R;
subplot(2,2,3);
imagesc(XI+YI);
colormap gray;
axis equal; axis off;
title('distorted grid');

%% distort image
Idistorted = interp2(X,Y,Icut,XI,YI,'linear');

%% plot distorted image
subplot(2,2,4);
imagesc(Idistorted);
grid on;
colormap gray;
title('distorted image'); 

