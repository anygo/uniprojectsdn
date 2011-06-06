% cleanup
close all;
clear all;
clc;

%load data
w_closing_connection = importdata('with_closing_connection')';
wo_closing_connection = importdata('without_closing_connection')';
java_rmi = importdata('java_rmi')';

% plot data
figure;
hold on;
plot(java_rmi(1,:), java_rmi(2,:), 'red');
plot(w_closing_connection(1,:), w_closing_connection(2,:), 'blue');
plot(wo_closing_connection(1,:), wo_closing_connection(2,:), 'blue--');
legend('Java RMI', 'RPC', 'RPC w/o closing connection', 'Location', 'NorthWest');
title('Java versus us');
xlabel('#messages');
ylabel('time [ms]');
grid on;
hold off;