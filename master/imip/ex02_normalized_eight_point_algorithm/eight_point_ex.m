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
%										 points in 2 images
% eight_point.m: in this file you compute the Fundamental
%					  matrix. Additionally, we implement the data 
%					  balancing of the input points. If the 
%					  Fundamental matrix is known, we can compute 
%                the epipolar line in the right image
%                for any point, we select in the left image
% NOTE: Complete the '???' lines!  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function eight_point_ex()
% Computes the Fundamental matrix and applies additionally 
% data balancing to the input points. Afterwards, the epipolar
% line in the right image is computed out of the selected point
% in the left image. Both epipolar lines (in the left and right 
% image) are shown.

close all;
clear all;

% Normalize the image data using isotropic scaling due to numerical
% instabilities
% 0: no
% 1: yes
ISO_SCALING = 1; 

% Initialize the 2 transformations T1 & T2
T1 = eye(3);
T2 = eye(3);

% Load the point correspondences of the two images:
load left_image_points;
load right_image_points;

% Size of the left image
[a b] = size(left_image_points);

% Number of corresponding points for the images
numberOfPoints = a;

% Copy the saved points into a working structure (array, ...)
points(:,1,1) = left_image_points(:, 1);
points(:,1,2) = left_image_points(:, 2);
points(:,2,1) = right_image_points(:, 1);
points(:,2,2) = right_image_points(:, 2);

% If balancing should be applied
if(ISO_SCALING)
   % Iterated over images: 
   % i=1 => left image
   % i=2 => right image
   for i=1:2
      % Create a 3x"Number of Point correspondences" zero matrix
      pts = zeros(3, numberOfPoints);
      % x-coordinate of the i-th image
      pts(1,:) = points(:,i,1);
      
      % y-coordinate of the i-th image
      pts(2,:) = points(:,i,2);
      
      % Homogeneous coordinate
      pts(3,:) = ones(1,numberOfPoints);
      
      % Create a 2x"Number of Point correspondences" matrix (m) to 
      % compute easier the mean
      % m: to compute mean of the x-/y-coordinates
      m = pts(1:2,:);    
      
      % (x,y) centroid of the points = 
      % sum along the second dimension of m =
      % translation of the image points
      % t: translation of the image points
      t = mean(m, 2);     
      
      % Centered points; mc is a 2x"Number of Point correspondences" matrix
      % mc: mean centered points
      mc = m - t*ones(1,numberOfPoints); 
      
      % Compute distance of each centered point to the origin (0,0)
      % dc is a 1x"Number of Point correspondences" vector
      % dc: distance to center
      % Formula: (x,y).^2 - (0,0).^2
      dc = sqrt(sum(mc.^2));    
      
      % Average distance (davg) to the origin
      davg = (1/numberOfPoints)*sum(dc); 
      
      % Scale factor (s), so that the average distance is sqrt(2)
      s = sqrt(2)/davg;     
      
      % Transformation matrix (T)
      T = [eye(2)*s, -s*t; 0 0 1];  
      
      % Save the transformation matrix for each image to denormalize the
      % data
      if(i == 1)
            % Transformation of the left image
            T1 = T;
      else
            % Transformation of the right image
            T2 = T;
      end
        
      % Apply the normalization transformation to the image points
      pts = T * pts;
      % Save back to the original structure
      points(:,i,1) = pts(1,:);
      points(:,i,2) = pts(2,:);

      % Check the facts of the normalization process
      % Average distance of the points of the i-th image to the origin
      g = sqrt(points(:,i,1).*points(:,i,1) + points(:,i,2).*points(:,i,2));
      avglength = (1/numberOfPoints).*(sum(sum(g)));
      % Origin, which should be approximately (0,0)
      originX = (1/numberOfPoints).*sum(sum(points(:,i,1)));
      originY = (1/numberOfPoints).*sum(sum(points(:,i,2)));
    end
end

A = zeros(numberOfPoints, 9);

% Compute the Measurement matrix A
for i=1:numberOfPoints
    x1 = points(i,1,1);
    y1 = points(i,1,2);
    x2 = points(i,2,1);
    y2 = points(i,2,2);
    
    A(i,:) = [x1*x2 y1*x2 x2 x1*y2 y1*y2 y2 x1 y1 1];
end

disp('cond(A)')
cond(A)

% Compute the SVD of the Measurement matrix A
[~, ~, V] = svd(A);

% Find Fundamental Matrix F: see nullspace
f = V(:, 9);

% ||f|| = 1; f is only defined up to scale; f = 0 should be not the
% solution!
disp('norm(f)')
nf = norm(f);

% if f has not length 1
for i=1:9
    f(i) = f(i)/nf;
end
F = reshape(f, 3, 3);

% Check rank criterion: rank(F) = 2
[FU FD FV]= svd(F);
FDnew = FD;
FDnew(3,3) = 0;

% Create Fundamental matrix FMM with rank 2
FMM = FU*FDnew*FV';
norm(FMM)

% Denormalize the data
% Apply the transformations T1 and T2
FM = T2'*FMM*T1; 
norm(FM)

% Save the Fundamental matrix FM into the current directory, 
% so you can use it again with the command load
save FM;

% Load the 2 images for plotting the epipolar lines
left_image = double(rgb2gray(imread('left2.jpg')));
right_image = double(rgb2gray(imread('right2.jpg')));

% Size of the left image
[~, t] = size(left_image);

figure(1);
imagesc(left_image); 
colormap(gray); 
title('Click a point on the left image');
axis image;

figure(2);
imagesc(right_image); 
colormap(gray); 
title('Corresponding epipolar line in the right image');
axis image;

% You can select 8 points for plotting the epipolar lines
list =['r' 'b' 'g' 'y' 'm' 'k' 'w' 'c'];

% Iterate over the 8 selected points
for i=0:7
   % Get a point on the left image    
   figure(1);    
   [left_x left_y] = ginput(1);
   hold on;
   plot(left_x,left_y,'r*');

   % Create a homogeneous point
   left_P = [left_x; left_y; 1];
   
   % Compute the corresponding point in the right image
   % with the Fundamental matrix FM
   right_P = FM*left_P;
   
   % Create samples for the x-coordinate of the epipolar line in the right
   % image
   right_epipolar_x = 1:t;

   % Use the equation of a line: ax + by + c = 0; 
   % y = (-c -ax) / b
   right_epipolar_y = (-right_P(3,1) - right_P(1,1)*right_epipolar_x) / right_P(2,1);
   
   % Plot epipolar line in the right image
   figure(2);
   hold on;
   plot(right_epipolar_x,right_epipolar_y,list(mod(i,8)+1));

   % Compute the epipolar line in the left image    
   % We know that left epipole is the 3rd column of V. 
   % FM = FU FD FV'
   [~, ~, FV]= svd(FM); % transformations
   
   left_epipole = FV(:,3);
   % Divide by the z-coordinate
   left_epipole = left_epipole/left_epipole(3);
    
   % Create samples for the x-coordinate of the epipolar line in the left
   % image
   left_epipolar_x = 1:t;
   left_epipolar_y = left_y + (left_epipolar_x - left_x) * ...
       (left_epipole(2) - left_y) / (left_epipole(1) - left_x);
   
   % Plot epipolar line in the left image
   figure(1);
   hold on;
   plot(left_epipolar_x,left_epipolar_y,list(mod(i,8)+1));
end