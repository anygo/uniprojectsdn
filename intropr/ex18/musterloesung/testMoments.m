function [restore] = testMoments(inIm)

close all 

if(nargin == 0)
    inIm = imread('momented.bmp');
else
    inIm = imread(inIm);
end
[nX,nY]=size(inIm);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Original Image
figure(1);
subplot(2,2,1);
imshow(inIm);
title('Original Image');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Zero-Order Moment
disp('Area: ');
m00 = zeroMoment(inIm)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   First-Order Moments
[m10, m01, m11] = firstMoments(inIm);

disp('Center of mass:' );
xCenter = round(m10/m00);
yCenter = round(m01/m00);
center = [xCenter; yCenter]

% Correct the position of the center
imgCenterX = round(nX/2);
imgCenterY = round(nY/2);

shiftIm = zeros(nX,nY);

for i = 1:nX
    for j = 1:nY
        % if the value 
        newX = i - xCenter + imgCenterX;
        newY = j - yCenter + imgCenterY;
        if(newX <= 0 || newY <= 0 || newX > nX || newY > nY)
            ;
        else
            shiftIm(newX,newY) = inIm(i,j);
        end
    end
end

     
            
subplot(2,2,2);
imshow(shiftIm,[]);
title('Shifted Image');

[m10, m01, m11] = firstMoments(shiftIm);

disp('Center of mass after shifting:' );
xCenter = round(m10/m00);
yCenter = round(m01/m00);
center = [xCenter; yCenter] %just check it if it works
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Second Order moments: Orientation

[m20,m02,m11] = secondMoments(shiftIm);

disp('Rotation of object: ')
rot = 0.5 * atan(2*m11 / (m20 - m02));
rotGrad = (rot * 360) / (2*pi)


%back transform, so exchange - at sin terms!
rotMat = [cos(rot), sin(rot); -sin(rot), cos(rot)];

rotIm = zeros(nX,nY);
for i = 1:nX
    for j = 1:nY
        % correct position by center, rotate, shift back
        newPos = (rotMat * [i-xCenter;j-yCenter]) + center;
        newX = round(newPos(1));
        newY = round(newPos(2));

        if(newX <= 0 || newY <= 0 || newX > nX || newY > nY)
               ; 
            else
               rotIm(newX,newY) = shiftIm(i,j);
        end
    end
end

subplot(2,2,3);
imshow(rotIm,[]);
title('Rotated Image, Forward Transform');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Back Transform to adress the sampling problem!

rotImInv = zeros(nX,nY);
rotMatInv = [cos(rot), -sin(rot); sin(rot), cos(rot)];
for i = 1:nX
    for j = 1:nY
       % map the point to the image
       newPos = (rotMatInv * [i-xCenter;j-yCenter]) + center;
       newX = round(newPos(1));
       newY = round(newPos(2));
       
       if(newX <= 0 || newY <= 0 || newX > nX || newY > nY)
                   ; %set later missing parts to 50
                else
                   rotImInv(i,j) = shiftIm(newX,newY);
       end
    end
end

subplot(2,2,4);
imshow(rotImInv,[]);
title('Rotated Image, Inverse Transform');
