clear all;
close all;

A = double(imread('yu_fill.jpg'));
%A=rand(300,300);

% how many approximations -> how many ranks -> svd or rank()
[U S V] = svd(A);
n = size(nonzeros(S),1);

% compute k'th approximation; k < n
k = n;
B = S(1,1)*U(:,1)*V(:,1)'; % init
for i = 2:k
    B = B + S(i,i)*U(:,i)*V(:,i)'; % now just add
    subplot(1,3,1);
    colormap gray;
    imagesc(A);
    title('original image');
    subplot(1,3,2);
    imagesc(B);
    line = sprintf('approximation %i', i);
    title(line);
    subplot(1,3,3);
    image(B-A);
    title('diff');
    pause;
end

