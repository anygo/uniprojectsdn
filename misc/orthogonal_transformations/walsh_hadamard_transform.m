function walsh_hadamard_transform()
    close all;
    clear all;
    clc;

    % length of signal
    N = 16;
    
    % create a 1D signal f of length N
    x = -0.5:1/(N-1):0.5;
    f = sin(12.345*x) .* cos(67*x);
    f = 2*mat2gray(f)-1; % f now in [-1;1]
    subplot(2,2,1); plot(x,f); title('original signal f(x) \in [-1;1], x \in [0.5;0.5]'); axis([-0.5 0.5 -1 1]);
    
    % create Hadamard matrix of same size^2 (NxN)
    H_2 = [1 1; 1 -1];
    H_cur = H_2;
    subplot(2,2,2); imagesc(H_2); colormap gray; title('Hadamard matrix H_2'); axis image; pause(0.1);
    for i = 2:log(N)/log(2)
        H_cur = kronecker_product(H_cur, H_2);
        imagesc(H_cur); title(['Hadamard matrix H_{' num2str(2^i) '}']); axis image; pause(0.1);
    end
    
    % transform signal
    c = H_cur * f';
    subplot(2,2,4); stem(c); axis([1 N min(c(:)) max(c(:))]); title('WHT coefficients (feature vector c)'); 
    
    % reconstruct (approximated) original signal
    f_rec = H_cur\c; % == inv(H_cur)*c
    subplot(2,2,3); plot(x,f_rec); title('reconstructed signal (red curve = error \approx 0)'); axis([-0.5 0.5 -1 1]);
    hold on;
    plot(x,abs(f_rec-f'), 'r'); % error
    hold off;
end

function K = kronecker_product(A, B)
    K = zeros(size(A) .* size(B));
    for r = 1:size(A,1)
        for c = 1:size(A,2)
            submatrix = A(r,c) * B;
            step = size(B,1);
            K((r-1)*step+1 : r*step, (c-1)*step+1 : c*step) = submatrix;
        end
    end
end