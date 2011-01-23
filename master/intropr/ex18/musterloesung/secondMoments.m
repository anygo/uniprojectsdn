function [m20,m02,m11] = secondMoments(img)

if(nargin == 0)
    return;
end

[nX,nY] = size(img);


m20 = double(0);
m02 = double(0);
m11 = double(0);

%centered coordinates! conflicts with Matlab csys
%we have to correct the coordinates!
dx = round(nX*0.5);
dy = round(nY*0.5);

for i = 1:nY
    for j = 1:nX
        m20 = m20 + double(img(i,j)) * (i - dx)^2;
        m02 = m02 + double(img(i,j)) * (j - dy)^2;
        m11 = m11 + double(img(i,j)) * (i - dx) * (j - dy);
    end
end