clear all;
clc;

% compute lpc coefficients a1 and a2 of f(t) = [-2 0 1 -1 0 2]
A = [0 -2; 1  0; -1 1; 0 -1];
b = [   1;   -1;    0;    2];

a = pinv(A)*b;

% determine f's
f2 = [-2 0] * a;
f3 = [1  0] * a;
f4 = [-1 1] * a;
f5 = [0 -1] * a;
f6 = [2  0] * a;

orig = [-2 0 1 -1 0 2 f6];
est = [-2 0 f2 f3 f4 f5 f6];
plot(0:6, orig, 'b');
hold on;
plot(0:6, est, 'r');
hold off;