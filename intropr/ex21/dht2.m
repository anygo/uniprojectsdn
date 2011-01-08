function [approx, detailH, detailV, detailD] = dht2(signal)
%--------------------------------------------------------------------------
% 2D Discrete Haar Transform
%--------------------------------------------------------------------------

% definitions of Haar analysis high- and lowpass filter
lp = 1/sqrt(2)*[ 1 1];
hp = 1/sqrt(2)*[-1 1];

% ------------rows-------------
% 1D dht along the rows
% needs periodic substitution of signal by one element (half the kernel
% size)
sig = [signal signal(:,1)];

% compute the convolution with hp and lp
approximation = conv2(sig,lp,'valid');
detail = conv2(sig,hp,'valid');

% downsampling by factor 2
approximation = approximation(:,1:2:end);
detail = detail(:,1:2:end);

% ------------columns-------------
% 1D dht along the columns of the approximation
% needs periodic substitution of signal by one element (half the kernel
% size)

% -----------------
% Exercise: 
% insert your part here
% -----------------
% remove this part if done
approx = zeros(size(signal) ./ 2);
detailH = approx;
detailV = approx;
detailD = approx;
% -----------------


end
