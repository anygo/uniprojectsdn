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
approximationH = conv2(sig,lp,'valid');
detailH = conv2(sig,hp,'valid');

% downsampling by factor 2
approximationH = approximationH(1:2:end,1:2:end);
detailH = detailH(1:2:end,1:2:end);

% ------------columns-------------
% 1D dht along the columns of the approximation
% needs periodic substitution of signal by one element (half the kernel
% size)

sig = [signal; signal(1,:)];

% compute the convolution with hp and lp
approximationV = conv2(sig,lp','valid');
detailV = conv2(sig,hp','valid');

% downsampling by factor 2
approximationV = approximationV(1:2:end,1:2:end);
detailV = detailV(1:2:end,1:2:end);


% in beide richtungen
sig = [signal; signal(1,:)];
sig = [sig sig(:,1)];

lp_diag = 1/sqrt(2)*[1 0; 0 1];
hp_diag = 1/sqrt(2)*[-1 0; 0 1];

% compute the convolution with hp and lp
approximationD = conv2(sig,lp_diag,'valid');
detailD = conv2(sig,hp_diag,'valid');

% downsampling by factor 2
%approximationD = approximationD(1:2:end,1:2:end);
detailD = detailD(1:2:end,1:2:end);

approx = sqrt((approximationH.^2 + approximationV.^2));
end
