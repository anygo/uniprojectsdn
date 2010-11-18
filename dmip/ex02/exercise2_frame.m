%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Diagnostic Medical Image Processing (DMIP) 
% WS 2010/11
% Author: Attila Budai & Jan Paulus
% Exercise: Image Undistortion
% Framework
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all; 
clc 

%%%%%%%%%%%%%%%%%%%%%
% 1. Pre-processing %
%%%%%%%%%%%%%%%%%%%%%

%   1. Load undistorted image to generate distorted one
%   If you want to check easily if there is an error in your implementation,
%   try with an easy phantom image, e.g.
%
%   Phantom 1: Black rectangle with white rectangle inside:
%   I = zeros(256);
%   I(100:200,100:200)=1;
%
%   Phantom 2: Columnwise colors:
%   I = zeros(256);
%   for g=1:256
%       I(:,g)=g;
%   end

Iorig = imread('undistorted', 'jpg'); 

%   2. Normalize intensity values [0, 1]
I = mat2gray(Iorig);

%   3. Cut the (m x n) image -> quadratic image
minI = (min(min(size(I))));
I = imcrop(I, [1 1 minI-1 minI-1]);
%   instead you can also use:
%   I = I(1:minI,1:minI);

figure(1);
subplot(2,3,1);			
imagesc(I);
colormap(gray);
grid on
axis image
title('Cutted Image');

%   4. Generate a grid to sample the image 
[X,Y] = meshgrid(1:minI,1:minI);
%   instead you can also use:
%   [Y,X] = ndgrid(1:minI,1:minI)

%   5. Create an artificial distortion field (ellipsoidal)
%   a: spread among x-direction
%   b: spread among y-direction
%   d: (d/2) = maximal value at the radius boundary
%   R - max(max(R))*0.5: shifting the positive range to half positive/half
%   negative range
a = 3;
b = 9;
d = 12;
h = floor(minI/2);
R = d*sqrt(a*((X-h)/minI).^2 + b*((Y-h)/minI).^2);
R = R - max(max(R))*0.5;

figure(1);
subplot(2,3,2);			
imagesc(R);
colormap gray
axis image
axis off
title(['a=', int2str(a), ',b=', int2str(b), ',d=', int2str(d)]);

% Undistorted/corrected image points
XU = X;
YU = Y;

% Distorted image points = undistorted image points + artificial
% ellipsoidal distortion field
XD = X + R;
YD = Y + R;

%   6. Compute the distorted image
%   Resample the image I at (XD,YD) to create artificial image Idist
Idist = interp2(X,Y,I,XD,YD);
%   The function will return NaN for samplings outside the image. Set them to zero
Idist(isnan(Idist)) = 0;

figure(1);
subplot(2,3,3);	
imagesc(Idist);
colormap gray
grid on
axis image
title('Distorted Image');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2. Image Undistortion - Workflow %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   1. Number of lattice points (this only works for symmetric images)

nx = 8; 
ny = 8;

% Step size
sx = size(I,2);
sy = size(I,1);

fx = sx / nx;
fy = sy / ny;


%   Fill the distorted and undistorted matrices for the lattice points with
%   values 

XU2 = zeros(ny,nx); 
YU2 = zeros(ny,nx);
XD2 = zeros(ny,nx);
YD2 = zeros(ny,nx);

% Select nx,ny feature points (usually provided by tracking point 
% correspondences in the phantom during the calibration step).
% Because of the rounding, we work with approximative point 
% correspondences (XU2,XD2)
% floor: take left neighbor

for r = 1:ny 
    for c = 1:nx
        XU2(r,c) = XU(floor(r*fy),floor(c*fx)); 
        YU2(r,c) = YU(floor(r*fy),floor(c*fx));
        XD2(r,c) = XD(floor(r*fy),floor(c*fx));
        YD2(r,c) = YD(floor(r*fy),floor(c*fx));
    end
end

B = zeros(ny,nx);

figure(2);
colormap(gray);
% mesh(Z) draws a wireframe mesh using X=1:n & Y=1:m, where [m,n]=size(Z)
meshc(XU2,YU2,B); 
axis([0 300 0 300])
view([-90 90]) % view([-270 90])
hidden
grid off
title('Grid for sampling the undistorted (ideal) image');

figure(3);
colormap(gray);
meshc(XD2,YD2,B);
axis([0 300 0 300])
view([-90 90]) % view([-270 90])
hidden
grid off
% Direction: undistorted image -> distorted image
% Where do we have to sample to get the distorted image?
title('Grid for sampling the original image to get the distorted image (XU->XD)');


%   Compute the distorted points: be aware of the fact, 
%   that the artificial deformation takes place from the distorted to
%   undistorted; for creation we used the undistorted + deformation =
%   distorted

XD2 = XU2 - (XD2-XU2);
YD2 = YU2 - (YD2-YU2);




figure(4);
colormap(gray);
meshc(XD2,YD2,B);
view([-90 90]) % view([-270 90])
axis([0 300 0 300])
hidden
grid off;
% Direction: distorted image -> corrected undistorted image
% Where do we have to sample the distorted image to get the corrected undistorted image?
title('Grid for sampling the distorted image to get the corrected image (XD->XU)');

%   2. Polynom of degree d

% Polynom of degree d -> (d-1): extrema
% d=0: constant (horizontal line with y-intercept a_0 -> f(x)=a_0)
% d=1: oblique line with y-intercept a_0 & slope a_1 -> f(x)=a_0 + a_1 x
% d=2: parabola
% d>=2: continuous non-linear curve 
% E.g. d=5: 4 extrema
d = 5;


%   Number of Coefficients
NumKoeff = (d+2)*(d+1)/2; % kleiner Gauss

%   Number of Correspondences
NumCorresp = nx*ny;

disp('Polynom of degree');
d
disp('NumKoeff');
NumKoeff
disp('NumCorresp'); 
NumCorresp

%   3. Create the matrix A
A = zeros(NumCorresp,NumKoeff);

% Realign the grid matrix to a vector.
% Use the colon operator
XU2vec = XU2(:);
YU2vec = YU2(:);
XD2vec = XD2(:);
YD2vec = YD2(:);

% Compute matrix A
for r = 1:NumCorresp
  c = 1;
  for i = 0:d
    for j = 0:(d-i)
      A(r,c) = YU2vec(r)^j*XU2vec(r)^i;
      c = c + 1;
    end
  end
end

% Compute the pseudo-inverse of A
[U, S, V] = svd(A);
disp('rank(A)');
rank(A)

Si = S';
epsilon = 1e-5;

for i = 1:size(S,2)
    if(S(i,i) < epsilon)
        Si(i,i) = 0;
    else
        Si(i,i) = 1/Si(i,i);
    end
end

Apseudoinv  = V*Si*U';

%   Compute the distortion coefficients u_i,j, v_i,j

Uvec = Apseudoinv * XD2vec;
Vvec = Apseudoinv * YD2vec;

%   4. Compute the XDist and YDist grid points which are used to sample the
%   distorted image to get the undistorted image
% (x,y) is the position in the undistorted image and (XDist,YDist) the
% position in the distored (observed) X-ray image. 


XDist = zeros(sy, sx);  
YDist = zeros(sy, sx);

for y = 1:sy 
    for x = 1:sx       
     c = 1;
        for i = 0:d
            for j = 0:(d-i)
                XDist(y,x) = XDist(y,x) + Uvec(c) * YU(y,x)^j * XU(y,x)^i;
                YDist(y,x) = YDist(y,x) + Vvec(c) * YU(y,x)^j * XU(y,x)^i;
                c = c + 1;
            end
        end
    end
end

%   5. Resample the distorted image at (XDist,YDist) to obtain the undistorted
%   image using bilinear interpolation

undist = interp2(X,Y,Idist,XDist,YDist);
undist(isnan(undist)) = 0;


figure(1);
subplot(2,3,4);			
imagesc(undist);
grid on
axis image
title('Undistorted/corrected Image');

figure(1);
subplot(2,3,5);			
%imagesc(abs(I-undist));
imagesc(log(1+abs(I-undist)));
%colorbar
axis image
title('Cutted - Corrected Image');

figure(1);
subplot(2,3,6);	
%imagesc(abs(I-Idist));
imagesc(log(1+abs(I-Idist)));
axis image
title('Cutted - Distorted Image');
%colormap('Jet');

%print -deps 'image_distortion.eps'
%print -djpeg 'image_distortion.jpeg'

disp('mean(mean(abs(xray-undist)))'); 
mean(mean(abs(I-undist)))
