function WaveletTest()

close all;

% create an image with a power of two size
imagesize = 256;
% creates a Shepp-Logan phantom image (used often in medical image
% processing) with intensities [0, 255]
img = phantom(imagesize) * 255; 

% you may add noise here for experiments
% noise = randn(imagesize, imagesize) * 0.05;
% img = img + noise;

% plot original image
figure;
subplot(2,2,1);
imagesc(img);
title('Original');
colormap gray;
axis image;

% compute 2D dht of the image
[approx, detailH, detailV, detailD] = dht2(img);

% stitch the four blocks with wavelet coefficients to one image
% |-------------------|
% | approx  | detailH |
% |-------------------|
% | detailV | detailD |
% |-------------------|
waveletcoeffs = [approx detailH; detailV detailD];

% visualize first decomposition
subplot(2,2,2);
imagesc(waveletcoeffs);
title('Wavelet-Koeffizienten')
colormap gray;
axis image;

% compute inverse 2D dht
recon = idht2(approx, detailH, detailV, detailD);

% plot reconstructed image and the difference
subplot(2,2,3)
imagesc(recon);
title('Rekonstruktion')
colormap gray;
axis image;
subplot(2,2,4)
imagesc(recon-img);
title('Differenz zum Original')
colormap gray;
axis image;


% execute the multiresolution analysis filterbank with 3 stages 
ndec = 3;
waveletcoeffs = mra2(img, ndec);

%----------------------------------------------------
%   COMPRESSION
%

% ---------------
% Exercise: 
% compute a compression scheme and determine the compression rate
% ---------------

%----------------------------------------------------

% plot the complete decomposition result
figure;
imagesc(waveletcoeffs);
colormap gray;
axis image;


% execute the multiresolution synthesis filterbank with 3 stages on the
% modified coefficients (after compression)
recon = mrs2(waveletcoeffs, ndec);
figure;
imagesc(recon);
colormap gray;
axis image;
title(['compression = ' num2str(compression)]);

% compute the mean quadratic error between the original and the
% reconstructed (compressed) image
mse = sum((recon(:)-img(:)).^2)/length(img(:));

% plot the difference image
figure;
imagesc(recon-img);
colormap gray;
axis image;
title(['difference, mse = ' num2str(mse)]);

end