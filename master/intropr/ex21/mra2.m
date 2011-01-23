function waveletcoeffs = mra2D(signal, ndec)
%--------------------------------------------------------------------------
% 2D Multiresolution Analysis Filterbank with Haar Wavelets
% ndec: number of decompositions
% ndec of 0 corresponds to a wavelet decomposition with the signal itself
%--------------------------------------------------------------------------

% initialize wavelet coefficients with signal
waveletcoeffs = signal;

sizey = size(waveletcoeffs,1);
sizex = size(waveletcoeffs,2);

for i = 1:ndec
    [approx, detailH, detailV, detailD] = dht2(waveletcoeffs(1:sizey,1:sizex));
    waveletcoeffs(1:sizey,1:sizex) = [approx detailH; detailV detailD];
    
    sizey = sizey/2;
    sizex = sizex/2;
end
