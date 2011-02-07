function walsh_hadamard_transform()
    close all;
    clear all;
    clc;

    % create a 1D signal f of length 128
    x = -64:63;
    f = sin(0.123*x) + cos(0.4*x);
    subplot(1,3,1); plot(x,f); title('original signal (f)');
    
    % create Hadamard matrix of same size (128x128)
    H_2 = [1 1; 1 -1];
    subplot(1,3,2); imagesc(H_2); colormap gray; title('Hadamard matrix H_2'); axis image; pause(0.5); 
    H_4 = kronecker_product(H_2, H_2); imagesc(H_4); title('Hadamard matrix H_4'); axis image; pause(0.5);
    H_8 = kronecker_product(H_4, H_2); imagesc(H_8); title('Hadamard matrix H_8'); axis image; pause(0.5);
    H_16 = kronecker_product(H_8, H_2); imagesc(H_16); title('Hadamard matrix H_{16}'); axis image; pause(0.5);
    H_32 = kronecker_product(H_16, H_2); imagesc(H_32); title('Hadamard matrix H_{32}'); axis image; pause(0.5);
    H_64 = kronecker_product(H_32, H_2); imagesc(H_64); title('Hadamard matrix H_{64}'); axis image; pause(0.5);
    H_128 = kronecker_product(H_64, H_2); imagesc(H_128); title('Hadamard matrix H_{128}'); axis image; pause(0.5);
    
    % transform signal
    c = H_128 * f';
    subplot(1,3,3); stem(c); axis([1 128 min(c(:)) max(c(:))]); title('transformed signal (feature vector c)');
    
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