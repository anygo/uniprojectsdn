function [recon] = idht2(approx, detailH, detailV, detailD)
%--------------------------------------------------------------------------
% 2D Inverse Discrete Haar Transform
%--------------------------------------------------------------------------

% definition of the Haar synthesis (reconstruction) high- and lowpassfilter
rlp = 1/sqrt(2)*[1  1];
rhp = 1/sqrt(2)*[1 -1];

% ------------rows-------------
% combine approx and detailH to get approx (higher resolution)
% upsample the coefficients by factor 2 along the rows 
usapprox = zeros(size(approx,1)*2, size(approx,2));
usapprox(2:2:end,:) = approx;

usdetail = zeros(size(detailH,1)*2, size(detailH,2));
usdetail(2:2:end,:) = detailH;

% periodic boundary condition by one element
usapprox = [usapprox; usapprox(1,:)];
usdetail = [usdetail; usdetail(1,:)];

% compute convolution with hp and lp and add both parts
approx = conv2(usapprox,rlp','valid') + conv2(usdetail,rhp','valid');

% combine detailV and detailD to get detail (higher resolution)
% upsample the coefficients by factor 2 along the rows 
usapprox = zeros(size(detailV,1)*2, size(detailV,2));
usapprox(2:2:end,:) = detailV;
usdetail = zeros(size(detailD,1)*2, size(detailD,2));
usdetail(2:2:end,:) = detailD;

% periodic boundary condition by one element
usapprox = [usapprox; usapprox(1,:)];
usdetail = [usdetail; usdetail(1,:)];

% compute convolution with hp and lp and add both parts
detail = conv2(usapprox,rlp','valid') + conv2(usdetail,rhp','valid');

% ------------columns-------------
% combine results of row section by upsampling and convolution to get the
% reconstruction

% upsample the coefficients by factor 2 along the columns
usapprox = zeros(size(approx,1), size(approx,2)*2);
usapprox(:,2:2:end) = approx;
usdetail = zeros(size(detail,1), size(detail,2)*2);
usdetail(:,2:2:end) = detail;

% periodic boundary condition by one element
usapprox = [usapprox usapprox(:,1)];
usdetail = [usdetail usdetail(:,1)];

% compute convolution with hp and lp and add both parts
recon = conv2(usapprox,rlp,'valid') + conv2(usdetail,rhp,'valid');

end
