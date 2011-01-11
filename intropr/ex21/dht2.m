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

approximation = [approximation; approximation(1,:)];
detail = [detail; detail(1,:)];

approx = conv2(approximation,lp','valid');
approx = approx(1:2:end,:);

detailH = conv2(approximation,hp','valid');
detailH = detailH(1:2:end,:);

detailV = conv2(detail,lp','valid');
detailV = detailV(1:2:end,:);

detailD = conv2(detail,hp','valid');
detailD = detailD(1:2:end,:);

end
