%% cleanup
%close all;
clear all;
clc;

%% (normalisiertes) histogramm erstellen
I = imread('object.png');
I = mat2gray(I);
bins = 128;

[counts x] = imhist(I,bins);
normalized_hist = counts / sum(counts);


%% maximum likelihood estimation
squared_error = ones(bins,1)*-1;
for theta_index = 1:bins
    params1 = mle(x(1:theta_index), 'frequency', counts(1:theta_index));
    params2 = mle(x(theta_index+1:end), 'frequency', counts(theta_index+1:end));
    
    params1(isnan(params1)) = 0;
    params2(isnan(params2)) = 0;
    if params1(2) == 0 || params2(2) == 0
        continue;
    end
    
    prior1 = sum(counts(1:theta_index))/sum(counts);
    prior2 = sum(counts(theta_index+1:end))/sum(counts);
    
    y1 = prior1 * eval_gauss(x, params1(1), params1(2));
    y2 = prior2 * eval_gauss(x, params2(1), params2(2));
    combined = y1 + y2;
    
    squared_error(theta_index) = sum((combined' - normalized_hist*sum(combined)).^2);
    
    cla;
    subplot(2,1,1);
    area(x, normalized_hist*sum(combined), 'FaceColor', 'black');
    hold on;
    %area(x, y1, 'FaceColor', 'r');
    %area(x, y2, 'FaceColor', 'r');
    area(x, combined, 'FaceColor', 'c');
    scatter(x(theta_index), 0, 'vblack');
    alpha(0.5);
    hold off;
    
    I2 = im2bw(I, x(theta_index));
    subplot(2,1,2), imagesc(I2), colormap gray;
    drawnow;
end

squared_error(squared_error == -1) = max(max(squared_error));

[tmp idx] = sort(squared_error);

theta = x(idx(1));

params1 = mle(x(1:idx(1)), 'frequency', counts(1:idx(1)));
params2 = mle(x(idx(1)+1:end), 'frequency', counts(idx(1)+1:end));

prior1 = sum(counts(1:idx(1)))/sum(counts);
prior2 = sum(counts(idx(1)+1:end))/sum(counts);

y1 = prior1 * eval_gauss(x, params1(1), params1(2));
y2 = prior2 * eval_gauss(x, params2(1), params2(2));
combined = y1 + y2;

subplot(2,1,1);
area(x, normalized_hist*sum(combined), 'FaceColor', 'black');
hold on;
%area(x, y1, 'FaceColor', 'r');
%area(x, y2, 'FaceColor', 'r');
area(x, combined, 'FaceColor', 'c');
scatter(theta, 0, 'vblack');
alpha(0.5);
hold off;
I2 = im2bw(I, theta);
subplot(2,1,2), imagesc(I2), colormap gray;