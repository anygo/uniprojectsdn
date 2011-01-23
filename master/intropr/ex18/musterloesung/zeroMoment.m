function [m00] = zeromoment(img)

% For an image, the moment of order zero is just the sum over all
% intensities

m00 = sum(sum(img));

%[nX,nY] = size(img);
% m00 = double(0);
% 
% for i = 1:nX
%     for j = 1:nY
%         m00 = m00 + double(img(i,j));
%     end
% end