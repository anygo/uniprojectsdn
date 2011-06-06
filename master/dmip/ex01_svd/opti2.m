clear all;
close all;

% all b's in one matrix
b = [1 1; -1 2; 1 -3; -1 -4]; 

% form M-matrix similar to the one in the lecture -> is that right???
M = zeros(size(b,1), 4);
M(:,1) = b(:,1).^2;
M(:,2) = b(:,1).*b(:,2);
M(:,3) = b(:,1).*b(:,2);
M(:,4) = b(:,2).^2;

% compute svd
[U S V] = svd(M);

% fill matrix (values are in V) -> is that right???
A = zeros(2,2);
A(1,1) = V(4,1);
A(1,2) = V(4,2);
A(2,1) = V(4,3);
A(2,2) = V(4,4);