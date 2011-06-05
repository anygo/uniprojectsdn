% cleanup
close all;
clear all;
clc;

%load data
rmi = importdata('rmi')';
rpc = importdata('rpc')';

% plot data
figure;
hold on;
plot(rmi(1,:), rmi(2,:), 'b--');
plot(rpc(1,:), rpc(2,:), 'r');
legend('Java RMI', 'RPC', 'Location', 'NorthWest');
title('Java versus us');
xlabel('#messages');
ylabel('time [ms]');
hold off;