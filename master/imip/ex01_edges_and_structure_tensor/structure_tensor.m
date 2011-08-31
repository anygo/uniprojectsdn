%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Interventional Medical Image Processing (IMIP)
% SS 2011
% (C) University Erlangen-Nuremberg
% Author: Attila Budai & Jan Paulus
% Exercise 1.4
% Structure tensor
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all;
clear all;
clc;

% 0: Rectangle
% 1: Fingerprint2
% 2: MR brain image
% 3: Fingerprint3
% 4: straight line for testing gradient, edge orientation, ...

fall = 2;

if(fall == 0)
    %im = zeros(200,200);
    %im(50:100,50:100) = 1;
    
    % Values for visual inspection
    im = zeros(10,10);
    im(5,5:8) = 1;
    
    maskSize = 3; 
    sigma = 1; 
    rho = 2; 

    % Threshold necessary for structure tensor
    thres = 0.001; % Rectangle
elseif(fall == 1)
    im = imread('fingerprint2.tif');
    im = double(im);

	maskSize = 13; 
    sigma = 14; 
    rho = 1; 

    % Threshold necessary for structure tensor
    thres = 0.0001; 
elseif(fall == 2)
    % Load the image
    filename = 'mr14.dcm';
    im = double(dicomread(filename)); % double!
    dicominfo(filename)

    maskSize = 23; % 3, 23
    sigma = 15;
    rho = 5;
    
    % Threshold necessary for structure tensor
    thres = 0.01;
elseif(fall == 3)
    im = imread('fingerprint3.tiff');
    im = double(im);

	maskSize = 13; 
    sigma = 0.5;
    rho = 4;

    % Threshold necessary for structure tensor
    thres = 190;
else
    im = zeros(10,10);
    im(5,5:7) = 1;

    maskSize = 3; 
    sigma = 1; 
    rho = 2; 

    % Threshold necessary for structure tensor
    thres = 0.0005;
end

% Size of the image
[m,n] = size(im);

% Maximum gray value of the image
max(max(im))

% Create evaluation grid positions
msh = floor(maskSize/2);
[Y,X] = ndgrid(-msh:msh, -msh:msh);

gaussMaskSi = (1./(2*pi*(sigma.^2))).*exp(-0.5.*(X.^2+Y.^2)./(sigma.^2));
min(min(gaussMaskSi))
max(max(gaussMaskSi))
DoGMask = -(X./(2.*pi.*(sigma.^4))).*exp(-0.5.*(X.^2+Y.^2)./(sigma.^2)); 
min(min(DoGMask))
max(max(DoGMask))
% Derivative in x direction
fx = conv2(im, DoGMask);
% Derivative in y direction
fy = conv2(im, DoGMask');

% Create figure using grayscales
figure(1);
colormap(gray);

% plot derivative in x direction
subplot(2,3,1);
imagesc(fx);
axis image
title('f_x');
xlabel('x');
ylabel('y');

% plot derivative in y direction
subplot(2,3,2);
imagesc(fy);
axis image
title('f_y');
xlabel('x');
ylabel('y');

% Matrix elements we will later use to build the tensor matrix
Jxx = fx .* fx;
Jyy = fy .* fy;
Jxy = fx .* fy;

gaussMaskrho = (1./(2*pi*(rho.^2))).*exp(-0.5.*(X.^2+Y.^2)./(rho.^2));

% Perform the Gaussian regularization
% Spatial averaging of the tensor elements J
Qxx = imfilter(Jxx, gaussMaskrho);               
Qyy = imfilter(Jyy, gaussMaskrho);               
Qxy = imfilter(Jxy, gaussMaskrho);      

% Initializations

% Corner matrix
cor = zeros(m,n);
% Matrix for homogeneous regions
homo = zeros(m,n);
% Matrix for edge elements
edg = zeros(m,n);

% Structure tensor matrix
part = zeros(2,2);
% Matrix for the angle plotting
angle = zeros(m,n);

% Matrix for eigenvalue lambda1
e1 = zeros(m,n);
% Matrix for eigenvalue lambda2
e2 = zeros(m,n);

for s=1:m
    for t=1:n
        % Build the structure tensor matrix (variable: part(4x4)) iteratively
        part(1,1) = Qxx(msh+s, msh+t);
        part(1,2) = Qxy(msh+s, msh+t);
        part(2,1) = Qxy(msh+s, msh+t);
        part(2,2) = Qyy(msh+s, msh+t);
             
        % Compute eigenvalues as a diagonal matrix D and
        % eigenvectors in a matrix V (both in increasing order!) of the
        % structure tensor
        [V D] = eig(part);
     
        e1(s,t) = D(2,2);
        e2(s,t) = D(1,1);
        
        
        % Discriminate the different cases: homogeneous region, edge, corner         
        if(e1(s,t) < thres && e2(s,t) < thres)
            %disp('homogeneous region');
            homo(s,t) = 1;
        elseif(e1(s,t) > e2(s,t) && e2(s,t) < thres)
            %disp('edge');
            edg(s,t) = 1;
        elseif(e1(s,t) >= e2(s,t) && e2(s,t) >= thres)
            %disp('corner');
            cor(s,t) = 1;
        end
       
        % angle between y-axis and edge orientation!
        % x = [0; 1]; 
        % angle between x-axis and edge orientation!
        x = [1; 0];
        % Access eigenvector corresponding to the lowest eigenvalue
        ev = V(:,1);
        % Computation of the angle between the eigenvector of the smallest
        % eigenvalue and the x-axis (1,0)
        denominator = norm(x)*norm(ev);
        nominator = x' * ev;
        angle(s,t) = acosd(nominator / denominator);
    end
end

subplot(2,3,4);
imagesc(cor);
axis image
title('Corners');
xlabel('x');
ylabel('y');
colorbar

subplot(2,3,5);
imagesc(homo);
axis image
title('Homogeneous regions');
xlabel('x');
ylabel('y');
colorbar

subplot(2,3,6);
imagesc(edg);
axis image
title('Edges');
xlabel('x');
ylabel('y');
colorbar

figure(2);
imagesc(angle);
colormap hsv
axis image
title('Angle');
xlabel('x');
ylabel('y');
colorbar

figure(4);
subplot(1,2,1);
imagesc(e1);
title('\lambda_1');
axis image
colorbar
subplot(1,2,2);
imagesc(e2);
title('\lambda_2');
axis image
colorbar
