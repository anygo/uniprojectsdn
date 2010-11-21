function test_targets = DifferentNorms(train_patterns, train_targets, test_patterns, params)
N = size(test_patterns,2);

class0 = train_targets == 0;
class1 = train_targets == 1;

% mean vectors for both classes
mu0 = mean(train_patterns(:, class0),2);
mu1 = mean(train_patterns(:, class1),2);

cov0 = cov(train_patterns(:, class0)');
cov1 = cov(train_patterns(:, class1)');


test_targets    = zeros(1,N); 

if params == 1 %L1-norm
    for i = 1:N
        x = test_patterns(:,i);
        dist0 = norm(x-mu0,1);
        dist1 = norm(x-mu1,1);
        if dist0 < dist1
            test_targets(1,i) = 0;
        else
            test_targets(1,i) = 1;
        end
    end
end

if params == 2 %L2-norm
    for i = 1:N
        x = test_patterns(:,i);
        dist0 = norm(x-mu0,2);
        dist1 = norm(x-mu1,2);
        if dist0 < dist1
            test_targets(1,i) = 0;
        else
            test_targets(1,i) = 1;
        end
    end
end

if params == 3 %Mahalanobis-distance
    cov0_inv = inv(cov0);
    cov1_inv = inv(cov1);
    for i = 1:N
        x = test_patterns(:,i);
        dist0 = (x-mu0)'*cov0_inv*(x-mu0);
        dist1 = (x-mu1)'*cov1_inv*(x-mu1);
        if dist0 < dist1
            test_targets(1,i) = 0;
        else
            test_targets(1,i) = 1;
        end
    end
end