%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Interventional Medical Image Processing (IMIP) 
% SS 2011
% Author: Attila Budai & Jan Paulus
% Exercises: Matlab introduction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Delete all used variables
% ???
% Close all applications 
close all;
% Clear screen
clc;

% save session to file log.txt
diary log.txt 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Part 1
% Basics
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Display text or array
disp('row vector'); 
disp('v1 = [ 1 2 3 ];');
% Create a row vector
v1 = [ 1 2 3 ];
v1

disp('column vector');
% Create a column vector
% ???


% Save workspace variables v1 v2 on disk
save mysession v1 v2
clear all;
% Load workspace variables v1 v2 from disk
load mysession

disp('m1 = [ 1 2 3; 4 5 6; 7 8 9 ];');
% Create matrix
m1 = [ 1 2 3; 4 5 6; 7 8 9 ];
m1

disp('determinant of m1');
% Determinant of matrix m1
det(m1)

% Create matrix only with zero entries (20x20)
% ???
% Create matrix only with one entries (10x10)
% ???
% Copy entries of m3 to m2, centered 
m2(6:15,6:15) = m3; 

disp('transpose of m1');
% Transpose matrix m1
% ???

disp('inverse of m1');
% Inverse matrix m1
inv(m1)

disp('m1(2:3,1:2)');
% Access specific elements in (row, column) format
m1(2:3,1:2)
disp('note: indices start with 1');
disp('centered element');
% ???

disp('m1(1:end,2)');
% Using the <end> index that is available for each vector or matrix
m1(1:end,2)

% Using the <:> returns all elements in that dimension 
disp('m1(1,:)');
m1(1,:)
disp('m1(:)');
m1(:)

% Size of matrix m1
% ???
% ???
% Size of vector v1
size(v1)

% Classical matrix multiplication
disp('mMult = m1 * m1;');
mMult = m1 * m1;
mMult

% Element-wise multiplication
disp('mMult = m1 .* m1;');
mMult = m1 .* m1;
mMult
disp('same holds for the / operator');

% Cast to other datatypes
v1i = int16(v1);
disp('Task: check datatype of the v1i variable in the Workspace.');
disp('cast v1 to uint32');

v3 = [1.5 2.9 3.2];
% Round elements towards minus infinity
% ???

% Simple operations (sum, mean, max, median, min, std, var) 
m1
disp('mean(m1)');
mean(m1)
disp('max(m1)');
max(m1)
disp('maximum of matrix m1');
max(max(m1))

% Compute SVD e.g. to compute pseudo inverse of a matrix
A = [ 11 10 14; 12 11 -13; 14 13 -66 ];
disp('svd(A)');
[ U S V ] = svd(A);
U
S
V
disp('U*S*V^{T}');
U*S*V'

disp('note: try to avoid using for statements in Matlab. They are slow!');
% Matrix with random numbers
R = rand(10,10);
Rcopy = R;
% Talk: delete all numbers smaller 0.5
% Serialize index from (row,col) to (i)
for i=1:size(R(:),1)     
    if(R(i)<0.5)
        R(i) = 0.0;
    end
end

disp('check');
disp('using the min(R) command');
% Return row vector of smallest elements for each column
min(R) 
disp('using the min(min(R)) command');
% Apply the min again on the min(R) to get the overall smallest element in R
min(min(R))

% Faster, better and same result
% Copy back original random matrix
R = Rcopy; 
disp('R(R(:)<0.5) = 0.0;');
R(R(:)<0.5) = 0.0;
disp('using the min(R) command');
% Return row vector of smallest elements for each column
min(R) 
disp('using the min(min(R)) command');
% Apply the min again on the min(R) to get the overall smallest element in R
min(min(R))

% Display a matrix as an image
% 2x2 figure, upper left index
% ???
colormap(gray);
% Scale data and display as image
% ???

% Load a CT slice phantom image
P = phantom(64); 
subplot(2,2,2); imagesc(P)

subplot(2,2,3); 
% Plot some 1D data
% Vector with elements ranging from -1 to 1 with stepsize 0.01
samples = -1:0.01:1;
% Evaluate sin function using the samples
mySignal = sin(2*pi*samples); 
% Plot with point style
plot(samples, mySignal, '.' );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Part 2
% Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Put this code into a new .m file and name the file flipMatrix.m
% ##### snipp ##########
% function [Hfl] = flipMatrix( H )
% % input: matrix H, output: matrix Hfl
%   Hfl = flipud(H);
% return;
% #######################
% you can call this function via
Hflip = flipMatrix(R);
disp('flip matrix');
Hflip
% Helptext of the function
help flipMatrix

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Part 3
% Tools
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Read/write data
%path = 'c:';
fileId = 1;
%filename = sprintf('%s/test%d.data', path, fileId);
filename = sprintf('test%d.data', fileId);
fidOUT = fopen( filename, 'wb', 'ieee-le' );
if(~fidOUT) 
    disp('Could not open file for writing.');
    filename
end

% By default all types are in double precision
fwrite(fidOUT, R, 'double'); % 'float', 'int16', ...
fclose(fidOUT);

fidOUT = fopen( filename, 'rb', 'ieee-le' );
if(~fidOUT) 
    disp('Could not open file.');
    filename
end
 
N = size(R,1)*size(R,2);
Rnew = fread( fidOUT, N, 'double' );
fclose(fidOUT);
disp('Rnew');
size(Rnew)   
Rnew = reshape(Rnew,size(R,1),size(R,2));
Rnew

filename='mr12.dcm';
% Read DICOM image
% ???
im = double(im);
% Get information about the dicom file
% ???
figure(4); 
subplot(2,3,1);
imagesc(im); 
colormap('gray'); 
% Show color coding
% ???

filename='mr12.png';
% Read same image as non-dicom file
% ???
im = double(im);
subplot(2,3,2); 
imagesc(im);
 
% Use another colormap
colormap('spring'); 
colorbar
% Display pixel values and distances interactively
% ???


% Resize image using bilinear interpolation
imR=imresize(im, 0.5, 'bilinear');
subplot(2,3,3);
imagesc(imR); 
colormap('gray');
% Title of the figure
% ???
% Label of the x-axis
% ???
% Label of the y-axis
ylabel('y');

% Create special 2D filters
h=fspecial('sobel');
% Multidimensional image filtering
% replicate: Input array values outside the bounds of the array are assumed
% to equal the nearest array border value
% ???
subplot(2,3,4);
imagesc(imF);

% 2D convolution
% 3x3 convolution mask
mask = (1/9).*[1 1 1; 1 1 1; 1 1 1];
% ???
subplot(2,3,5);
imagesc(imCon);

si=size(im)
% Generate array for multidimensional functions and interpolation
% meshgrid: X and Y array for 3D plots
% !!! [X,Y]=meshgrid(1:si(1),1:si(2));
% Generate array for N dimensional functions
[Y,X] = ndgrid(1:si(1),1:si(2));
imY=im([2:end,end],:)-im(:,:);
imX=im(:,[2:end,end])-im(:,:);
magnitude = sqrt(imX.^2+imY.^2);
subplot(2,3,6);
imagesc(magnitude);
title('gradient');

[YE,XE]=ndgrid(1:200,1:200);
imE=im(1:200,1:200);
imEY=imE([2:end,end],:)-imE(:,:);
imEX=imE(:,[2:end,end])-imE(:,:);

figure(6);
% quiver or velocity plot
% ???
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Part 4
% External C++ functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Copy this part into a .c file and compile it via
% mex timestwo.c at the MATLAB prompt.
% After compilation it can be used then inside MATLAB via
% x = 2;
% y = timestwoalt(x)

% /*
%  * =============================================================
%  * timestwoalt.c - example found in API guide
%  *
%  * Use mxGetScalar to return the values of scalars instead of 
%  * pointers to copies of scalar variables.
%  *
%  * This is a MEX-file for MATLAB.
%  * Copyright (c) 1984-2000 The MathWorks, Inc.
%  * =============================================================
%  */
%  
% /* $Revision: 1.1.4.9 $ */
% 
% #include "mex.h"
% 
% void timestwoalt(double *y, double x)
% {
%   *y = 2.0*x;
% }
% 
% 

% % % The parameters
% % % nlhs 
% % %    MATLAB sets nlhs with the number of expected mxArrays. 
% % % 
% % % plhs 
% % %    MATLAB sets plhs to a pointer to an array of NULL pointers. 
% % % 
% % % nrhs 
% % %    MATLAB sets nrhs to the number of input mxArrays. 
% % % 
% % % prhs 
% % %    MATLAB sets prhs to a pointer to an array of input mxArrays. 
% % %    These mxArrays are declared as constant; they are read only and should 
% % %    not be modified by your MEX-file. Changing the data in these mxArrays may produce undesired side effects.
   
% void mexFunction(int nlhs, mxArray *plhs[],
%                  int nrhs, const mxArray *prhs[])
% {
%   double *y;
%   double x;
% 
%   /* Create a 1-by-1 matrix for the return argument. */
%   plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
% 
%   /* Get the scalar value of the input x. */
%   /* Note: mxGetScalar returns a value, not a pointer. */
%   x = mxGetScalar(prhs[0]);
% 
%   /* Assign a pointer to the output. */
%   y = mxGetPr(plhs[0]);
%   
%   /* Call the timestwoalt subroutine. */
%   timestwoalt(y,x);
% }

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Part 5
% Homework: Browse through the Matlab help
% and have a look on the different additional toolboxes 
% that might be available (depending on the license).
% This is for sure worth to do it and can save you later
% a lot of time. (Use: Help->Matlab help or press F1
% E.g. Image Processing Toolbox.  
% or use >> demo
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


