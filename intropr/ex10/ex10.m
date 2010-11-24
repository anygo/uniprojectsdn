function ex10()
close all;
clear all;
clc;

I = imread('object.png');

bins = 128;
[counts x] = imhist(I,bins);

totalSquaredError = zeros(bins,1);

for theta = 2:bins-1
    
    % f <= theta
    vec_left = [];
    for i = 1:theta
        vec_left = [vec_left; repmat(x(i),counts(i),1)];
    end
    left = mle(vec_left);
    squaredErrorLeft = 0;
    
    if sum(counts(1:theta)) > 0
        for i = 1:theta
            squaredErrorLeft = squaredErrorLeft + (eval_gauss(x(i), left(1), left(2)) - counts(i)/sum(counts(1:theta)))^2;
        end
    else
        totalSquaredError(theta) = -1;
    end
    
    
    
    % f > theta
    vec_right = [];
    for i = theta+1:bins
        vec_right = [vec_right; repmat(x(i),counts(i),1)];
    end
    right = mle(vec_right);
    squaredErrorRight = 0;
    
    if sum(counts(theta+1:bins)) > 0
        for i = theta+1:bins
            squaredErrorRight = squaredErrorRight + (eval_gauss(x(i), right(1), right(2)) - counts(i)/sum(counts(theta+1:bins)))^2
        end

        totalSquaredError(theta) = squaredErrorLeft + squaredErrorRight;
    else
        totalSquaredError(theta) = -1;
    end
    
   
end
end