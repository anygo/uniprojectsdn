%% clean up
%close all;
clear all;
clc;

%% ex 22
w = [-1 1 0 2; ... 
     -2 3 1 0.5];
 
mu = mean(w,2);
w2 = w - repmat(mu, 1, size(w,2));
[U S V] = svd(w2);

subplot(1,3,1);
hold on;
scatter(w(1,:), w(2,:));
axis([-5 5 -5 5]);
axis equal;
plot([mu(1,1) mu(1,1)+U(1,1)*S(1,1)], [mu(2,1) mu(2,1)+U(2,1)*S(1,1)]);
plot([mu(1,1) mu(1,1)+U(1,2)*S(2,2)], [mu(2,1) mu(2,1)+U(2,2)*S(2,2)]);
hold off;


%% ex 23a
w = [0 2  0 -2; ... 
     1 0 -1  0];
 
mu = mean(w,2);
w2 = w - repmat(mu, 1, size(w,2));
[U S V] = svd(w2);

subplot(1,3,2);
hold on;
scatter(w(1,:), w(2,:), 'r');
axis([-5 5 -5 5]);
axis equal;
plot([mu(1,1) mu(1,1)+U(1,1)*S(1,1)], [mu(2,1) mu(2,1)+U(2,1)*S(1,1)], 'r');
plot([mu(1,1) mu(1,1)+U(1,2)*S(2,2)], [mu(2,1) mu(2,1)+U(2,2)*S(2,2)], 'r');
hold off;


%% ex 23b
w = [2 4  2 0; ... 
     1 0 -1 0];
 
mu = mean(w,2);
w2 = w - repmat(mu, 1, size(w,2));
[U S V] = svd(w2);

subplot(1,3,3);
hold on;
scatter(w(1,:), w(2,:), 'g');
axis([-5 5 -5 5]);
axis equal;
plot([mu(1,1) mu(1,1)+U(1,1)*S(1,1)], [mu(2,1) mu(2,1)+U(2,1)*S(1,1)], 'g');
plot([mu(1,1) mu(1,1)+U(1,2)*S(2,2)], [mu(2,1) mu(2,1)+U(2,2)*S(2,2)], 'g');
hold off;