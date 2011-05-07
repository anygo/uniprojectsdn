%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Interventional Medical Image Processing (IMIP)
% SS 2011
% (C) University Erlangen-Nuremberg
% Author: Attila Budai & Jan Paulus
% Exercise 1.3: Differentiation as an ill-posed problem
% NOTE: Complete the '???' lines!  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
clc;

% Compute the noise free signal
x(1:20) = 0;

for i=21:39
    x(i)= i-20;
end
x(40:60) = 20;

% Plot the signal
figure(1);
subplot(2,3,1);
plot(x);
title('Original signal')
xlabel('x')
ylabel('f(x)')
axis([-10 70 -10 30])

% Compute and plot the first derivative
dx = conv(x, [1 -1], 'valid');
subplot(2,3,2);
plot(dx);
title('First derivative')
xlabel('x')
ylabel('f´(x)')
axis([-10 70 -0.8 1.8])

% Compute and plot the second derivative
ddx = conv(dx, [1 -1], 'valid');
subplot(2,3,3);
plot(ddx);
title('Second derivative')
xlabel('x')
ylabel('f´´(x)')
axis([-10 70 -0.8 0.8])

% Add Gaussian noise with \mu=0 and \sigma=0.9
xN = x + 0.1*randn(size(x));

% Plot the noisy signal
subplot(2,3,4);
plot(xN);
title('Original signal with gaussian noise \mu=0 and \sigma=0.9')
xlabel('x')
ylabel('f(x)')
axis([-10 70 -10 30])

% Compute and plot the first derivative of the noisy signal
dx = conv(xN, [1 -1], 'valid');
subplot(2,3,5);
plot(dx);
title('First derivative')
xlabel('x')
ylabel('f´(x)')
axis([-10 70 -0.8 1.8])

% Compute and plot the second derivative of the noisy signal
ddx = conv(dx, [1 -1], 'valid');
subplot(2,3,6);
plot(ddx);
title('Second derivative')
xlabel('x')
ylabel('f´´(x)')
axis([-10 70 -0.8 0.8])
