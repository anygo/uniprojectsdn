%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Interventional Medical Image Processing (IMIP)
% SS 2011
% (C) University Erlangen-Nuremberg
% Author: Attila Budai & Jan Paulus
% Exercise 2: Eight Point Algorithm, Fundamental matrix
% 				  and Epipolar geometry
% NOTES: 
% get_point_correspondences.m: with this file you can 
% 										 select the corresponding
%										 points alternating in 2 images
% eight_point.m: in this file you compute the Fundamental
%					  matrix. Additional, we implement the data 
%					  balancing of the input points. If the 
%					  Fundamental matrix is known, we can compute 
%                the epipolar line in the right image
%                for any point, we select in the left image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function get_point_correspondences()
% Loads 2 images and saves 10 point correspondences
% in variables: left_image_points and right_image_points

close all;
clear all;

% Load the 2 images and convert ithem from RGB to grayscale
left_image = double(rgb2gray(imread('left2.jpg')));
right_image = double(rgb2gray(imread('right2.jpg')));

% Number of point correspondences for the Eight Point Algorithm.
% You can try it with 8 or more points.
numberOfPoints = 10;

% Size of the left image
[m n] = size(left_image);

% Visualize the 2 images and click the point correspondences
figure(1);
imagesc(left_image); 
colormap(gray); 
title('Click a point on the left image'); 
axis image; 

figure(2); 
imagesc(right_image);
colormap(gray); 
title('Click the corresponding point on the right image'); 
axis image;

% Get the numberOfPoints point correspondences
for i=1:numberOfPoints
   figure(1);
   % Get the point from the left image
   [left_x left_y] = ginput(1);
   hold on; 
   plot(left_x, left_y,'rx'); 

	figure(2);
   % Get the corresponding point from the right image
   [right_x right_y] = ginput(1);
   hold on; 
   plot(right_x, right_y,'rx');
   
   % Save the point coordinates
   left_image_points(i,1) = left_x;
   left_image_points(i,2) = left_y;
   right_image_points(i,1)= right_x;
   right_image_points(i,2)= right_y;
   
   % Save them into the current directory
   save left_image_points;
   save right_image_points; 
end