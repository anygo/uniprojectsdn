function recon = mrs2D(waveletcoeffs, ndec)
%--------------------------------------------------------------------------
% 2D Multiresolution Synthesis Filterbank with Haar Wavelets
% ndec: number of decompositions
% ndec of 0 corresponds to a wavelet decomposition with the signal itself
%--------------------------------------------------------------------------

sizex = size(waveletcoeffs,1)/(2^ndec);
sizey = size(waveletcoeffs,2)/(2^ndec);

for i=ndec:-1:1

    % compute inverse dht
    waveletcoeffs(1:2*sizex,1:2*sizey) = idht2(waveletcoeffs(1:sizex,1:sizey),... %approx
        waveletcoeffs(1:sizex,sizey+1:2*sizey),... %detailH
        waveletcoeffs(sizex+1:2*sizex,1:sizey),... %detailV
        waveletcoeffs(sizex+1:2*sizex,sizey+1:2*sizey)); %detailD
    
    % after each iteration the number of approximation coefficients double
    sizex = sizex * 2;
    sizey = sizey * 2;
    
end

recon = waveletcoeffs;

end
