function [m10,m01,m11] = firstmoments(img)

% The first order moments give the center of mass in an image

if(nargin == 0)
    return;
end

[nX,nY] = size(img);

% in x-direction:
% go over each column, and add up the intensity weighted with the
% x-coordinate

% in y-direction
% go over each row, and add up the intensity weighted with the y-coordinate
% 


m10 = double(0);
m01 = double(0);
m11 = double(0);

for i = 1:nY
    for j = 1:nX
        m10 = m10 + double(img(i,j)) * i;
        m01 = m01 + double(img(i,j)) * j;
        m11 = m11 + double(img(i,j)) * i * j;
    end
end

