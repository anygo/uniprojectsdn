%% cleanup
%close all;
clear all;
clc;

%% ex 15
% g(x)
sigma = 1;
x = -5:10/499:5;
g_x = 1 / sqrt(2*pi*sigma^2) * exp(-0.5*x.^2/sigma^2);

subplot(1,2,1), plot(x, g_x);

% FT{g(x)}
N = 100;
k = -5:10/(N-1):5;
coeffs = 1 / sqrt(2*pi) * exp(-2*(pi*sigma*k).^2);

subplot(1,2,2), plot(k, coeffs);

% Fourier Series Approximation
approx_e = zeros(1,size(x,2));
approx_sin_cos = zeros(1,size(x,2));
for a = 1:size(x,2)
    xx = x(1,a);
    for n = 1:N
        approx_e(a) = approx_e(a) + coeffs(1,n) * exp(1i * n * xx); % this is exactly the same as
        approx_sin_cos(a) = approx_sin_cos(a) + coeffs(1,n) * (cos(n*xx) + 1i*sin(n*xx)); % this :-)
    end
    approx_e(a) = approx_e(a) / (2*pi);
    approx_sin_cos(a) = approx_sin_cos(a) / (2*pi);
end

subplot(1,2,1), hold on, plot(x, abs(approx_e), 'r'), hold off;