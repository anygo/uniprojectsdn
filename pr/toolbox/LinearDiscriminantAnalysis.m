function test_targets = LinearDiscriminantAnalysis(train_patterns, train_targets, test_patterns, params)

N = length(train_patterns);


class0 = find(train_targets == 0);
class1 = find(train_targets == 1);

%% training step
% mean vectors for both classes
mu0 = mean(train_patterns(:, class0)')';
mu1 = mean(train_patterns(:, class1)')';

% joint covariance matrix
covJoint = zeros(2,2);
for i = 1:N
    x = train_patterns(:,i);
    if train_targets(1,i) == 0
        covJoint = covJoint + (x-mu0)*(x-mu0)';
    else
        covJoint = covJoint + (x-mu1)*(x-mu1)';
    end
end

% compute SVD
[U S V] = svd(covJoint);

Phi = S^(-0.5)*U';

mu0_t = Phi*mu0;
mu1_t = Phi*mu1;

figure(3);
cla;
hold on;
for i = 1:N
   x = Phi*train_patterns(:,i);
   scatter(x(1,1), x(2,1));
end
title('transformed trainingset');
hold off;
figure(1);


%% classification step
test_targets = zeros(1,size(test_patterns,2));
log_p0 = log(size(class0,2)/N);
log_p1 = log(size(class1,2)/N);
for i = 1:size(test_patterns,2)
    x = test_patterns(:,i);
    x_t = Phi*x;
    prob0 = 0.5*norm(x_t-mu0_t) - log_p0;
    prob1 = 0.5*norm(x_t-mu1_t) - log_p1;
    if prob0 > prob1
        test_targets(1,i) = 0;
    else
        test_targets(1,i) = 1;
    end
end

end

