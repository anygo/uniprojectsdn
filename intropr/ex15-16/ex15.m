%% cleanup
close all;
clear all;
clc;

%% ex15
x = -5:0.1:5;
sigma = 1;
g_x = 1 / sqrt(2*pi*sigma^2) * exp(-0.5*x.^2/sigma^2);

plot(x, g_x);