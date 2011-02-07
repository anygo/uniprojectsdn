function haar_transform()
    close all;
    clear all;
    clc;

    % length of signal (must be a "power of 2")
    N = 128;
    
    % create a 1D signal f of length N
    x = 0:1/(N-1):1;
    f = sin(12.345*x) .* cos(67*x);
    f(1,N/2+1:end) = sin(100*x(1,N/2+1:end)) - cos(x(1,N/2+1:end).^2); 
    f(1,N/5:N/3) = 0.75;
    f = 2*mat2gray(f)-1; % f now in [-1;1]
    subplot(2,2,1); plot(x,f); title('original signal f(x) \in [-1;1], x \in [0;1]'); axis([0 1 -1 1]);
    
    % create Haar matrix of same size^2 (NxN)
    M = N; % see slides
    Har = create_haar_matrix(M);
    subplot(2,2,2); imagesc(Har); colormap gray; title(['Haar matrix Har_{' num2str(M) '}']); axis image;
    
    % transform signal
    c = Har * f';
    subplot(2,2,4); stem(c); axis([1 N min(c(:)) max(c(:))]); title('Haar coefficients (feature vector c)'); 
    
    % reconstruct (approximated) original signal
    f_rec = Har\c; % == inv(Har)*c
    subplot(2,2,3); plot(x,f_rec); title('reconstructed signal (red curve = error \approx 0)'); axis([0 1 -1 1]);
    hold on;
    plot(x,abs(f_rec-f'), 'r'); % error
    hold off;
end

function Har = create_haar_matrix(M)

    % see slides ;)
    Har = zeros(M, M);
    Har(1,:) = 1/sqrt(M); % k = 0
    for k = 1:M-1
        % compute p and q
        p = floor(log(k)/log(2)); 
        q = k - 2^p + 1;
        Har(k+1,:) = h(zeros(1,size(Har,2)),p,q);        
    end
    
    
    function x = h(x,p,q)
        x_vec = 0:1/(size(x,2)-1):1;

        x(1, ( x_vec >= (q-1)/2^p   ) & ( x_vec <= (q-0.5)/2^p) ) = 2^(p/2);
        x(1, ( x_vec >= (q-0.5)/2^p ) & ( x_vec <= q/2^p)       ) = -2^(p/2);
        x = 1/sqrt(M) * x;
    end
    
end