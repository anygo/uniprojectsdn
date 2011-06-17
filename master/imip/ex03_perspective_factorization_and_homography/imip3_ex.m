%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Interventional Medical Image Processing (IMIP)
% SS 2011
% (C) University Erlangen-Nuremberg
% Author: Attila Budai & Jan Paulus
% Exercise 3: Perspective Factorization & Homography                                             
% NOTE: Complete the '???' lines!                                       
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;

% Normalize the image data using isotropic scaling due to numerical
% instabilities
% 0: no
% 1: yes
ISO_SCALING = 1; 

noiseLevel = 0.09;
xmin = -1;
xmax = 1;
ymin = -1;
ymax = 1;

% create point correspondences
[X,Y] = meshgrid(-5.1:.5:4.9);
R = sqrt(X.^2 + Y.^2) + eps;
Z = 5 * sin(R)./R;
xc = X(:);
yc = Y(:);
zc = Z(:);

% create homogeneous coordinates
[m,n] = size(xc);
x4d = [xc(:), yc(:), zc(:), ones(m*n,1)]';

figure(1); 
plot3( xc(:), yc(:), zc(:), '.' ); 
xlabel('x');
ylabel('y');
zlabel('z');
grid on
drawnow;
title('3D points');

numberOfTrackedPoints = length(xc);
numberOfFrames = 20; 

P = zeros(numberOfFrames, 3, 4);
points = zeros(numberOfTrackedPoints, numberOfFrames, 2);

% (u0, v0) principal point 
u0 = 0.0;
v0 = 0.0;
f = 0.050; % focal length 50 mm = 0.05 m

px = 0.0004;
py = 0.0004;
sx = f/px;
sy = f/py;
s = 0;

KPR = [ sx s  u0 0;
        0  sy v0 0;
        0  0  1  0];

% rotation angle of each frame
angleStep = 9 / 180 * pi;
rotAngle = numberOfFrames*angleStep; % = pi -> 180 degree

cd = 1; 
number = 1;

% translation vector (x/y plane)
t1 =  0;
t2 =  0;
t3 =  800; % 700: more distance, 400: lower distance

% rotate camera around y-axis
alphay = 0;
alphaz = 0;
alphax = 0;

for alphay=0:angleStep:(rotAngle-angleStep)
    alphay;
    number;
    
    RotX = [ 1  0                 0;
             0  cd*cos(alphax)   cd*(-sin(alphax));
             0  cd*sin(alphax)   cd*cos(alphax)];
         
    RotY = [ cd*cos(alphay)  0   cd*(-sin(alphay));
             0               1   0;
             cd*sin(alphay)  0   cd*cos(alphay)];
         
    RotZ = [ cd*cos(alphaz)  cd*(-sin(alphaz))  0;
             cd*sin(alphaz)  cd*cos(alphaz)     0;
             0               0                  1];
               
    Rot = RotX * RotY * RotZ;     
    
    % extrinsic parameters (R,t)
    T = [t1 t2 t3]';
    R = [ Rot   T;
          0 0 0 1];
      
    % resulting 3x4 projection matrix = intrinsic * extrinsic
    Pt = KPR*R;
    
    % perspective projection of the points
    x2d = Pt * x4d;
    x = x2d(1,:) ./ x2d(3,:);
    y = x2d(2,:) ./ x2d(3,:);
    err = rand(size(x))-0.5;
    x(:) = x(:) + noiseLevel.* err(:);
    err = rand(size(x))-0.5;
    y(:) = y(:) + noiseLevel.* err(:);
    points(:,number,1) = x;
    points(:,number,2) = y;
    
    figure(2);
    subplot(2,2,1); 
    plot(points(:, number, 1), points(:, number, 2), 'ks') % o
    axis([xmin xmax ymin ymax])
    title('Perspective Factorization');
    xlabel('x');
    ylabel('y');
    grid on
    drawnow
    pause(0.5)
    P(number,:,:) = Pt;
    number = number + 1;    
end

% point correspondences are stored in:
% points(pointIndex, frameIndex, dimension)

if(ISO_SCALING) 
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % (1) Translation of the points that their dentroid is at the origin
    % (2) Scaling that the average distance from the origin is equal to
    %     sqrt(2)
    % (3) This transformation is applied to each of the images independently
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     
     for i=1:numberOfFrames
        i;
        pts = zeros(3, numberOfTrackedPoints);
        pts(1,:) = points(:,i,1);
        pts(2,:) = points(:,i,2);
        pts(3,:) = ones(1,numberOfTrackedPoints);
        xn = pts(1:2,:);            % xn is a (2xN) matrix
        N = size(xn,2);             % = numberOfTrackedPoints
        t = (1/N) * sum(xn,2);      % this is the (x,y) centroid of the points (2x1); sum along the second dimension
        xnc = xn - t*ones(1,N);     % center the points; xnc is a (2xN) matrix
        center(:,i,1) = xnc(1,:);
        center(:,i,2) = xnc(2,:);
        
        dc = sqrt(sum(xnc.^2));     % distance of each new calculated point to the origin (0,0); dc is (1xN) vector
        davg = (1/N)*sum(dc);       % average distance to the origin
        s = sqrt(2)/davg;           % the scale factor, so that average distance is sqrt(2)
        T = [s*eye(2), -s*t ; 0 0 1];   % transformation matrix
        pts = T * pts;
        points(:,i,1) = pts(1,:);
        points(:,i,2) = pts(2,:);
        
        a=sqrt(points(:,i,1).*points(:,i,1) + points(:,i,2).*points(:,i,2));
        avglength=(1/numberOfTrackedPoints).*(sum(sum(a)));
        
        originX=(1/numberOfTrackedPoints).*sum(sum(points(:,i,1)));
        originY=(1/numberOfTrackedPoints).*sum(sum(points(:,i,2)));
    end
end

% Part I  Perspective Factorization
translations = zeros(numberOfFrames, 2);
measurementMatrix = zeros(3*numberOfFrames, numberOfTrackedPoints);

% initialize projective depth with ones
lambda = ones(numberOfFrames, numberOfTrackedPoints);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (4) Compute for each frame the mean of image point coordinates
% (5) Subtract the mean from the image coordinates 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n = 1;
for i=1:numberOfFrames
     measurementMatrix(n,:) = points(:, i, 1);
     measurementMatrix(n+1,:) = points(:, i, 2);  
     measurementMatrix(n+2,:) = 1;
    n = n + 3;
end

% keep the original measurement matrix
W = measurementMatrix;

lc = 0.5;
lambdaChange = 1000;
oldLambda = 2000;

while(abs(lambdaChange-oldLambda) > lc) 
    %for lo = 1:5 % fixed number of iterations
        change = abs(lambdaChange-oldLambda);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % (7) Normalizing rows, use Frobenius-norm
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for i=1:numberOfFrames
             lambda(i,:) = lambda(i,:) / norm(lambda(i,:), 'fro');
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % (8) Normalizing columns, use Frobenius-norm 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        for i=1:numberOfTrackedPoints
            lambda(:,i) = lambda(:,i) / norm(lambda(:,i), 'fro');
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % (6) Build the measurement matrix
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        j = 1;
        for i=1:numberOfFrames
             measurementMatrix(j,:) = W(j,:) .* lambda(i,:);
             measurementMatrix(j+1,:) = W(j + 1,:) .* lambda(i,:);
             measurementMatrix(j+2,:) = lambda(i,:);
            j = j + 3;
        end
      
        cond(measurementMatrix)
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % (9) Compute the SVD
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         [U,S,V] = svd(measurementMatrix);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % (10) Find its nearest rank-4 approximation using the SVD
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         for i=5:size(S,1)
             S(i,i) = 0;
         end
        oldM = measurementMatrix;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % (11) Get the motion (camera movement)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Ssq = sqrt(S);
         Pfac = U * Ssq;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % (11) Get the structure (3D points)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         Xfac = Ssq * V';
        
        figure(3); 
        plot3(Xfac(1,:), Xfac(2,:), Xfac(3,:), '.'); 
        xlabel('x');
        ylabel('y');
        zlabel('z');
        grid on
        drawnow;
        title('Estimated 3D points');
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % (12) Reproject the 3D points into each image to obtain new 
        %      estimates of the depths
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        measurementMatrix = U * S * V';

        measurementMatrix = measurementMatrix ./ W;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % (13) Get new lambda's
        % (14) Compute the mean of these three new lambda's
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        j = 1;
        for i=1:numberOfFrames
            lambdaMean = 1/3 * ( measurementMatrix(j,:) + measurementMatrix(j+1,:) + measurementMatrix(j+2,:) );
            lambda(i,:) = lambdaMean;
            j = j + 3;
        end
        
        oldLambda = lambdaChange;
        lambdaChange = norm(measurementMatrix - oldM, 'fro');
        lambdaChange
    %end
end

sx = size(Xfac);
X3D = zeros( 3, sx(2) );
X3D(1,:) = Xfac(1,:) ./ Xfac(4,:);
X3D(2,:) = Xfac(2,:) ./ Xfac(4,:);
X3D(3,:) = Xfac(3,:) ./ Xfac(4,:);

% plot the projective reconstruction
figure(2);
subplot(2,2,2);
plot3(X3D(1,:), X3D(2,:), X3D(3,:), 'ks');
grid on
axis equal;
xlabel('x');
ylabel('y');
zlabel('z');
title('Projective reconstruction');

% Homography:
% The projective reconstruction does not give us a good 
% idea how the reconstructed object does look like. 
% Therefore we will jump from the projective reconstruction 
% to a metric reconstruction using ground truth.
% Fortunately we have given the real (not the computed)
% projection matrices. This will allow us to compute the
% HOMOGRAPHY between the projective and the metric
% reconstruction.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (16) Compute the homography
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x0 = eye(4,4);
xdata = Xfac([1:4],:); % projective reconstruction
ydata = x4d;
size(xdata)
size(ydata)
% solve non-linear curve-fitting problem in least-squares sense;
% given input data xdata;
% observed output data ydata;
% find coefficients h that best fit the equation
% non-linear function homography; @ means can be specified as a 
% function handle

% lsqcurvefit ???
[h, resnorm] = lsqcurvefit(@homography,x0,xdata,ydata);

% homography
h 
% value of the squared 2-norm of the residual at h
resnorm

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (17) Jump to the metric space
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
XMetric = h*xdata;
XMetric(1,:) = XMetric(1,:) ./ XMetric(4,:);
XMetric(2,:) = XMetric(2,:) ./ XMetric(4,:);
XMetric(3,:) = XMetric(3,:) ./ XMetric(4,:);

subplot(2,2,3);
plot3(XMetric(1,:), XMetric(2,:), XMetric(3,:), 'ks');
grid on
xlabel('x');
ylabel('y');
zlabel('z');
axis equal;
title('Metric reconstruction');