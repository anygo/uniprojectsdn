%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Diagnostic Medical Image Processing (DMIP) 
% WS 2010/11
% Author: Attila Budai & Jan Paulus
% Exercises: Matlab introduction - Speed
% 
% Implement two solutions for setting elements of
% a random matrix to zero if they are beneath a 
% threshold and compute (1/element) otherwise. 
% Have a look at the speed differences.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Clear screen and variables, close figures
%clc;
clear all;
close all;

% Used threshold
thr = 0.5;

% Create a large randomized matrix of size (dim1 x dim2);
dim1 = 4000;
dim2 = 3000;

A = rand(dim1, dim2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solution 1: Use loops
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Start timer
tic;

% Get matrix dimensions
[sy,sx] = size(A);
% Initialize resulting matrix with zeros
B = zeros(sy,sx);

% Start loops
for i=1:sy
    for j=1:sx
        if A(i,j)<thr
            B(i,j)=0;
        %else
         %   B(i,j)=1/A(1,j);
        end
    end
end

% Stop timer and show elapsed time
time = toc;
fprintf('Elapsed time with loops:\t %f seconds.\n', time);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solution 2: Avoid loops
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Start timer
tic;

% Again get matrix dimensions
[sy,sx] = size(A);
% and initialize resulting matrix with zeros
C = zeros(sy,sx);
% This time: Directly access elements
C(A(:,:)<thr) = 0;
%C(A(:,:)>=thr) = 1/(A(:,:)>=thr);

% Stop timer and show elapsed time
time = toc;
fprintf('Elapsed time without loops:\t %f seconds.\n', time);
