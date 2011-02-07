% cleanup
close all;
clear all;
clc;

% compute a quadratic decision boundary
% combined S1 and S2
S = [1 1; 1 2; 2 1; 3 1; 2 4; 2 3; 3 2; 4 2]';
% class labels
L = [0 0 0 0 1 1 1 1];

% a) compute coefficient vector a for the decision boundary

% "measurement" matrix
% [x^2 xy y^2]
M = zeros(size(S,2), 3);
M(:,1) = S(1,:)' .^ 2;
M(:,2) = S(1,:)' .* S(2,:)';
M(:,3) = S(2,:)' .^ 2;

% compute least squares solution
a = pinv(M)*L';

% plot the decision boundary
[Y X] = meshgrid(0:0.2:7, 0:0.2:7);
Z = a(1)*X.^2 + a(2).*X.*Y + a(3)*Y.^2;
contour(X, Y, Z, 0.5);
hold on;
scatter(S(1,L==1), S(2,L==1));
scatter(S(1,L==0), S(2,L==0), 'r');

% b) classifiy two features
scatter([2 6], [6 2], 'blackx');
hold off;


