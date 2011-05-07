function ex5()
    % cleanup
    clear all;
    close all;
    clc;

    % load GRI scores
    load('scores1.mat');
    load('scores2.mat');
    load('scores3.mat');
    load('targets.mat');

    % how many points to compute and plot?
    n_points = 100;
    
    % compute ROC-curve for classifier 1, 2 and 3
    points1 = generate_roc_curve(scores1, targets, n_points);
    points2 = generate_roc_curve(scores2, targets, n_points);
    points3 = generate_roc_curve(scores3, targets, n_points);
    
    % compute area using trapz
    area1 = abs(trapz(points1(1, :), points1(2, :))) - 0.5;
    area2 = abs(trapz(points2(1, :), points2(2, :))) - 0.5;
    area3 = abs(trapz(points3(1, :), points3(2, :))) - 0.5;
    
    % plot everything
    figure;
    hold on;
    plot(points1(1, :), points1(2, :), 'r');
    plot(points2(1, :), points2(2, :), 'g');
    plot(points3(1, :), points3(2, :), 'b');
    hold off;
    xlabel('fp rate');
    ylabel('tp rate');
    title('ROC-curves');
    legend(['Classifier 1: ' num2str(area1)], ... 'Classifier 2', 'Classifier 3', 'Location', 'SouthEast');
           ['Classifier 2: ' num2str(area2)], ...
           ['Classifier 3: ' num2str(area3)], ...
            'Location', 'SouthEast');
end

% generates points for ROC-curve
function points = generate_roc_curve(scores, targets, n_points)
    % array with 2D-points
    points = zeros(2, n_points);
    
    % compute stepwidth for threshold -> [0..1]
    threshs = -eps:1/(n_points-1):1;
    
    for i = 1:n_points
        [tprate, fprate] = compute_tp_and_fp_rate(scores, targets, threshs(i));
        points(2, i) = tprate;
        points(1, i) = fprate;
    end
end

% computes tp and fprate for a given threshold
function [tprate, fprate] = compute_tp_and_fp_rate(scores, targets, threshold)
    ground_truth_positives = sum(targets == 1);
    ground_truth_negatives = sum(targets == 0);
    
    true_positives = sum(targets == 1 & scores > threshold);
    false_positives = sum(targets == 0 & scores > threshold);
    
    tprate = true_positives/ground_truth_positives;
    fprate = false_positives/ground_truth_negatives;
end