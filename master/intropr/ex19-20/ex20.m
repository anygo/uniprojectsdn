clear all;
clc;

% compute lpc coefficients a1 and a2 of f(t) = [-2 0 1 -1 0 2]
A = [0 -2; 1  0; -1 1; 0 -1];
b = [   1;   -1;    0;    2];

a = pinv(A)*b;

% determine f's
% don't know which 'direction' 
f2a = [-2 0] * a;
f3a = [1  0] * a;
f4a = [-1 1] * a;
f5a = [0 -1] * a;
f6a = [2  0] * a;

f2 = [0 -2] * a;
f3 = [0  1] * a;
f4 = [1 -1] * a;
f5 = [-1 0] * a;
f6 = [0  2] * a;

% plot
orig = [-2 0 1 -1 0 2 f6];
est = [-2 0 f2 f3 f4 f5 f6];
est2 = [-2 0 f2a f3a f4a f5a f6a];
plot(0:6, orig, 'b');
hold on;
plot(0:6, est, 'r');
plot(0:6, est2, 'g');
hold off;