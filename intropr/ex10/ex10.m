%% cleanup
close all;
clear all;
clc;

%% (normalisiertes) histogramm erstellen
I = imread('object.png');
I = mat2gray(I);
bins = 512;

[counts x] = imhist(I,bins);
normalized_hist = counts / sum(counts);
counts(1)=0;


%% maximum likelihood estimation
for theta = 1:bins
    params1 = mle(x(1:theta), 'frequency', counts(1:theta));
    params2 = mle(x(theta+1:end), 'frequency', counts(theta+1:end));
    
    prior1 = sum(counts(1:theta))/sum(counts);
    prior2 = sum(counts(theta+1:end))/sum(counts);
    
    cla;
    %plot(x, normalized_hist, 'black');
    hold on;
    y1 = prior1 * eval_gauss(x, params1(1), params1(2));
    y2 = prior2 * eval_gauss(x, params2(1), params2(2));
    plot(x, y1, 'red');
    plot(x, y2, 'cyan');
    line([x(theta) x(theta)], [0 5]);
    hold off;
    
    drawnow;
end