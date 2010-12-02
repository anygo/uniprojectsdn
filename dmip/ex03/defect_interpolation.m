%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Diagnostic Medical Image Processing (DMIP) 
% WS 2009/10
% Author: Christopher Rohkohl, Attila Budai & Jan Paulus
% Exercise: Defect Pixel Interpolation
% NOTE: Complete the '???' lines
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Used variables:
% g: observed image
% w: defect binary pixel mask (0=defect, 1=okay)
% f: discrete signal
% maxit: number of iterations
% pad: size for image padding

% Defect Pixel Interpolation example
function defect_interpolation

    clear all;  
    close all;
    clc;
    
    % Number of iterations
    maxIter = 4000; 
    
    % Load test image (discrete signal f)
    I = double(imread('testimg.bmp')); 

    [m,n] = size(I); % m=112, n=118
    
    if (m<n)
        minS = m;
    else
        minS = n;
    end

    if (mod(minS,2) == 0)
        disp('even')
    else
        disp('odd')
        minS = minS - 1;
    end
    
    %minS
    
    cutI = I(1:minS,1:minS);
    
    % Load the defect pixel mask (binary window w)
    % 1=ok, 0=defect
    load('mask');
    
    % Observed image g = f*w (this is not Matlab notation)
    g = cutI.*w;
    
    figure(1);
    subplot(3,4,1); 
    imagesc(cutI);
    colormap gray
    axis image
    title('Original cutted image');
    
    subplot(3,4,2); 
    imagesc(g);
    axis image
    title('Original image with defect pixels');
    
    % Spatial domain: Median filtering
    medI = medfilt2(g);
    
    subplot(3,4,3); 
    imagesc(medI);
    axis image
    title('Reconstructed image (spatial domain)');
    
    subplot(3,4,4); 
    imagesc(abs(cutI-medI)); 
    title('|Original - spatial interpolated|');
    % colorbar
    axis image
   
    imgrec = interpdefectimage(cutI, g, w, maxIter, 32);
    
    subplot(3,4,7); 
    imagesc(imgrec); 
    title('Reconstructed image (frequency domain)');
    axis image
    
    subplot(3,4,12); 
    imagesc(imgrec-medI); 
    title('Difference between reconstructed images (frequency-spatial)');
    axis image
    colorbar
end

function f = interpdefectimage(im, g, w, maxit, pad)
    % input: ideal image im, observed image g, defect binary pixel mask w, maximal
    % iterations maxit, size for padding pad
    % output: reconstructed image f
    
    % Hint: Increase the image dimension to increase the resolution in the
    % frequency domain by padding the image g and mask w. Keep in mind that the number 
    % of computed fourier coefficients depends on the number of (pixel) samples
    % in the spatial domain. To increase the number of computed fourier coefficients (frequency resolution)
    % we can simply increase the image size by padding. 
    odim = size(g);
    g = padarray(g, size(g), 'symmetric', 'post');
    w = padarray(w, size(w), 'symmetric', 'post');
    
    % Image dimension
    dim = size(g);      
    % Half of the image dimension
    halfDim = floor(dim/2); 
    % Fourier Transform of g & w
    G = fft2(g);
    W = fft2(w);
    
    maxDeltaE_G_Ratio = Inf;
    maxDeltaE_G_Ratio_Tres = 1.0e-6;

    % Initialization
    Fhat = zeros(dim);
    FhatNext = zeros(dim);
    
    % Iterations
    for i=1:maxit
        
        % Check convergence criterion
        if (maxDeltaE_G_Ratio <= maxDeltaE_G_Ratio_Tres)
            sprintf('maxDeltaE_G_Ratio: %.15f', maxDeltaE_G_Ratio)
            break;
        end

        % In the i-th iteration select the line pair s1,t1
        % which maximizes the energy reduction [Paragraph after Eq. (16) in the paper]
        deltaE_G = abs(G(1:dim(1), 1:halfDim(2)+1));
        % Find the indices of the maximum values & return them in the
        % vector idx
        [maxDeltaE_G] = max(max(deltaE_G));
        idx = find(deltaE_G==maxDeltaE_G);
        % Return a random permutation of the integers 1:(Number of elements
        % in array)
        r = randperm(numel(idx));
        % Subscripts from linear index [I,J] = ind2sub(siz,IND)
        [s1, t1] = ind2sub(dim, idx(r(1)));
        
        % Calculate the ratio of energy reduction
        % in comparison to the last iteration
        if (i > 1) 
            maxDeltaE_G_Ratio = abs((maxDeltaE_G - lastEredVal(i-1))/maxDeltaE_G);
        end

        % Save the last energy reduction value
        lastEredVal(i) = maxDeltaE_G; 
        
        s1f = s1 - 1; 
        t1f = t1 - 1; 
        
        % Compute the corresponding linepair s2,t2:
        % mirror the positions at halfDim
        s2 = s1; t2 = t1;
        
        if s1f > 0
            s2 = halfDim(1) - (s1 - halfDim(1)) + 2;
        end
        if t1f > 0
            t2 = halfDim(2) - (t1 - halfDim(2)) + 2;
        end
        if s1f==0 && t1f==0 % Special case (0,0)
            s2 = -1; t2 = -1;
        end
        
        % This we require in the next step:
        % M = mod(X,Y) if Y ~= 0, returns X - n.*Y where n = floor(X./Y)
        % [Paragraph after Eq. (17) in the paper]
        twice_s1 = mod(2*s1f,dim(1))+1;
        twice_t1 = mod(2*t1f,dim(2))+1;
        
        % Estimate the new Fhat (FhatNext)
        % 4 special cases, where only a single line can be selected:
        % (0,0),(0,M/2),(N/2,0),(N/2,M/2))
        specialCases = [0 0; 0 halfDim(2); halfDim(1) 0; halfDim(1) halfDim(2)];
        % B = repmat(A,m,n) creates a large matrix B consisting of an
        % m-by-n tiling of copies of A. The size of B is [size(A,1)*m, (size(A,2)*n]
        % 0 becomes 1, numbers > 0 becomes 0
        hilfsmat = specialCases == repmat([s1f t1f], 4, 1);
      
        hilfsmat = sum(hilfsmat,2) == 2;

        if any(hilfsmat)
            % Handle any of the special cases: (0,0), (0,M/2), (N/2,0),
            % (N/2,M/2)
            FhatNext(s1, t1) = FhatNext(s1, t1) + (dim(1)*dim(2))*G(s1,t1)/W(1,1);
            fprintf('SPECIAL case \n');
        else 
            % Handle the general case
            tval = dim(1)*dim(2) * (G(s1,t1)*W(1,1) - conj(G(s1,t1))*W(twice_s1,twice_t1)) / ...
                (abs(W(1,1))^2 - abs(W(twice_s1,twice_t1))^2);
            
            % Accumulation: Update pair (s1,t1),(s2,t2)
            FhatNext(s1, t1) = FhatNext(s1, t1) + tval;
            FhatNext(s2, t2) = FhatNext(s2, t2) + conj(tval);
        end

        % End iteration step by forming the new error spectrum
        G = G - convhelper(FhatNext-Fhat, W, s1, t1);
        % Make sure we don't get any rounding errors,
        % G(s1,t1) and G(s2,t2) should be zero
        G(s1,t1) = 0;
        if all([s2,t2]~=-1) 
            G(s2,t2) = 0; 
        end
        Fhat = FhatNext;
        
        % Visualize each 100. iteration
        if (mod(i,100)==0)            
            fhat = real(ifft2(Fhat)); % Obtain the estimation of the real image
            idx = find(w==0);         % Find the important mask entries
            f = g;
            f(idx) = fhat(idx);
            f = f(1:odim(1),1:odim(2));
            
            subplot(3,4,5);
            imagesc(fhat);
            axis image
            title('Approximated image (fhat)');
            
            subplot(3,4,7);
            imagesc(f);
            axis image
            title('Reconstructed image (frequency domain)');
            
            subplot(3,4,8); 
            imagesc(abs(im-f)); 
            title('|Original - frequency interpolated|');
            %colorbar
            axis image
            xlabel('x');
            ylabel('y');

            subplot(3,4,6);
            clims = [ 0 1000 ];
            imagesc(abs(G), clims); 
            axis image
            title('Error spectrum: abs(G)');
            
            subplot(3,4,10);
            clims = [ 0 1000 ];
            imagesc(abs(ifftshift(G)), clims);  
            axis image
            title('Error spectrum: abs(ifftshift(G))');
            
            subplot(3,4,9);
            plot(1:i, log(lastEredVal(1:i)/dim(1) +(abs(lastEredVal(1:i))==0)));
            title('Energy');
            xlabel('Number of iterations');
            
            drawnow;
            %pause
        end
    end
    
    % Compute the inverse fourier transform of the estimated image 
    fhat = real(ifft2(Fhat)); % Obtain the estimation of the real image
    idx = find(w==0);         % Find the important mask entries
    f = g;                    % Return the result
    f(idx) = fhat(idx);
    f = f(1:odim(1),1:odim(2));
end

% Do the convolution of the m-times-n matrix F and W
% s,t is the position of the selected
% line pair, the convolution is simplified
% in the following way:
% G(k1,k2) = F(k1,k2) 'conv' W(k1,k2) 
%          = (F(s,t)W(k1-s,k2-t) + F*(s,t)W(k1+s,k2+t)) / (MN)
% where F* is the conjugate complex.

function G = convhelper(F, W, s, t)
    sz = size(F);
    G = zeros(sz);
    sf = s-1;
    tf = t-1;
    F_st = F(s,t);
    F_stc = conj(F_st);
    
    % sz(1): number of rows, sz(2): number of columns
    [X,Y] = meshgrid(0:sz(1)-1, 0:sz(2)-1);
    I = sub2ind(sz, X(:)+1, Y(:)+1);
    I_neg = sub2ind(sz, mod(X(:)-sf,sz(1))+1, mod(Y(:)-tf,sz(2))+1);
    I_pos = sub2ind(sz, mod(X(:)+sf,sz(1))+1, mod(Y(:)+tf,sz(2))+1);
    
    if (all(I_neg==I_pos))
        % Special case
        G(I) = F_st.*W(I_neg) ./ prod(sz);
    else
        G(I) = (F_st.*W(I_neg) + F_stc.*W(I_pos)) ./ prod(sz);
    end
end

