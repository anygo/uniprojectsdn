% cleanup
close all;
clear all;
clc;

% polynomial curve fitting
x = -25:0.5:25;
polynomial_degree = 6;

% do the work
y = zeros(size(x));
ground_truth_coeffs = rand(1,polynomial_degree+1)*10 - 5; % in [-5;5]
for d = 0:polynomial_degree
    y = y + ground_truth_coeffs(d+1) * x.^d;
end

% add noise
noise = ones(1,size(x,2)) + randn(1,size(x,2))/2;
y = y + mean(abs(y))*noise;
hold on; scatter(x,y,'x'); hold off;

% compute measurement matrix M
M = ones(size(x,2), polynomial_degree+1);
for d = 0:polynomial_degree
    M(:,d+1) = x.^d;
end

% estimate using pseudo inverse
estimated_coeffs = (pinv(M) * y')';

% plot estimated curve
x2 = min(x(:)):0.1:max(x(:));
y2 = zeros(size(x2));
for d = 0:polynomial_degree
    y2 = y2 + estimated_coeffs(d+1) * x2.^d;
end

hold on; plot(x2,y2, 'r'); hold off;