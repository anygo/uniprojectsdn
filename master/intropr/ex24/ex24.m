%% cleanup
%close all;
clear all;
clc;


%% visualize original data
load('pca_data.mat');
subplot(2,2,1);
cla;
hold on;
scatter(patterns(1, classes == 0), patterns(2, classes == 0), 'ro');
scatter(patterns(1, classes == 1), patterns(2, classes == 1), 'bo');
axis equal;
hold off;


%% perform pca
% translate every vector such that mean == (0,0)
mu = mean(patterns, 2);
tmp_patterns = patterns - repmat(mu, 1, size(patterns, 2));

% compute SVD
[U S V] = svd(tmp_patterns);
e1 = U(:,1); lambda1 = sqrt(S(1,1));
e2 = U(:,2); lambda2 = sqrt(S(2,2));

% visualize eigenvectors
subplot(2,2,1);
hold on;
plot([mu(1) mu(1)+e1(1)*lambda1], [mu(2) mu(2)+e1(2)*lambda1], 'black');
plot([mu(1) mu(1)+e2(1)*lambda2], [mu(2) mu(2)+e2(2)*lambda2], 'black');
hold off;

subplot(2,2,3);
hold on;
plot([mu(1) mu(1)+e1(1)*lambda1], [mu(2) mu(2)+e1(2)*lambda1], 'black');
plot([mu(1) mu(1)+e2(1)*lambda2], [mu(2) mu(2)+e2(2)*lambda2], 'black');
axis equal;
hold off;


%% transform (project on eigenvector with largest eigenvalue)
projected_2d = zeros(size(patterns));
projected_1d = zeros(1, size(patterns, 2));
for i = 1:size(patterns, 2)
    projected_1d(:,i) = e1' * patterns(:,i);
    projected_2d(:,i) = projected_1d(:,i) * e1;
end

projected_2d = projected_2d + repmat(mu, 1, size(projected_2d, 2));

% plot projected 2d-vectors
hold on;
scatter(projected_2d(1, classes == 0), projected_2d(2, classes == 0), 'r.');
scatter(projected_2d(1, classes == 1), projected_2d(2, classes == 1), 'b.');
hold off;

% compute remaining stuff
mu0 = mean(projected_1d(classes == 0));
mu1 = mean(projected_1d(classes == 1));

x = min(projected_1d):0.1:max(projected_1d);
counts0 = hist(projected_1d(classes == 0), x);
counts1 = hist(projected_1d(classes == 1), x);

% and a mle assuming a gaussian for each class
params0 = mle(x, 'frequency', counts0);
params1 = mle(x, 'frequency', counts1);

% visualize 1d crap
subplot(2,2,2);
cla;
hold on;
axis([min(x) max(x) 0 max([counts0 counts1])*1.2]);
area(x, counts0, 'FaceColor', 'b');
area(x, counts1, 'FaceColor', 'r');
plot(mu0, max([counts0 counts1])*1.1, 'bx');
plot(mu1, max([counts0 counts1])*1.1, 'rx');
text(mu0+0.05, max([counts0 counts1])*1.1, 'mean0');
text(mu1+0.05, max([counts0 counts1])*1.1, 'mean1');
alpha(0.7);
set(gca,'YTickLabel',[]);
set(gca,'YTick',[]);
hold off;

% visualize mle-gaussian of the two classes
x_gauss = min(projected_1d):0.01:max(projected_1d);
y_gauss0 = eval_gauss(x_gauss, params0(1), params0(2));
y_gauss1 = eval_gauss(x_gauss, params1(1), params1(2));
subplot(2,2,4);
cla;
hold on;
axis([min(x_gauss) max(x_gauss) 0 max([y_gauss0 y_gauss1])*1.2]);
area(x_gauss, y_gauss0, 'FaceColor', 'b');
area(x_gauss, y_gauss1, 'FaceColor', 'r');
plot(mu0, max([y_gauss0 y_gauss1])*1.1, 'bx');
plot(mu1, max([y_gauss0 y_gauss1])*1.1, 'rx');
text(mu0+0.05, max([y_gauss0 y_gauss1])*1.1, 'mean0');
text(mu1+0.05, max([y_gauss0 y_gauss1])*1.1, 'mean1');
alpha(0.7);
set(gca,'YTickLabel',[]);
set(gca,'YTick',[]);
hold off;


%% classify patterns (ex 24c)
a = [0.4; 0.6];
b = [-0.2; -0.6];

% transform to 1d
a_1d = e1' * a;
b_1d = e1' * b;

if norm(a_1d - mu0) < norm(a_1d - mu1)
    disp('1st test vector belongs to class 0');
else
    disp('1st test vector belongs to class 1');
end

if norm(b_1d - mu0) < norm(b_1d - mu1)
    disp('2nd test vector belongs to class 0');
else
    disp('2nd test vector belongs to class 1');
end
