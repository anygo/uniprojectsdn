%% cleanup
close all
clear all
clc

%% params
dim = 3;
n_features = 2500;

%% shift around 0-vector
f = [0 0 0; 0 1 1; 2 1 1; -2 -1 3; 1 0 -1]';
%f = randn(dim, n_features) + ones(dim,n_features)*5;
f_mean = mean(f,2);
f = f - repmat(f_mean,1,size(f,2));

%scatter3(f(1,:),f(2,:),f(3,:),'x');
%axis([-5 5 -5 5 -5 5]);

hold on;
[U S V] = svd(f);

for i = 1:dim
    plot3([0 U(1,i)*sqrt(S(i,i))],[0 U(2,i)*sqrt(S(i,i))],[0 U(3,i)*sqrt(S(i,i))],'r');
end

for i = 1:size(f,2)
    pt = U(:,1)'*f(:,i)*U(:,1);
    pt = pt + U(:,2)'*f(:,i)*U(:,2);
    %plot3([f(1,i) pt(1)],[f(2,i) pt(2)],[f(3,i) pt(3)]);
    scatter3(pt(1),pt(2),pt(3),'.','magenta');
end
hold off;
pause(1);

cla;
hold on;
for i = 1:size(f,2)
    pt = U(:,1:2)'*f(:,i);
   % pt = pt + U(:,2)'*f(:,i);
    %plot3([f(1,i) pt(1)],[f(2,i) pt(2)],[f(3,i) pt(3)]);
    scatter(pt(1),pt(2),'.','magenta');
end
hold off;