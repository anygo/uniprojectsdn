function test_targets = MaximumLikelihood(train_patterns, train_targets, test_patterns, approach)


%% initialization
L			= length(train_targets);
N               = size(test_patterns, 2);
test_targets    = zeros(1,N); 


%% compute mean vectors
mu0 = [0; 0];
mu1 = [0; 0];
mu0_count = 0;
mu1_count = 1;
for i = 1:L
    if (train_targets(1,i) == 0)
        mu0_count = mu0_count+1;
        mu0 = mu0 + train_patterns(:,i);
    else
        mu1_count = mu1_count+1;
        mu1 = mu1 + train_patterns(:,i);
    end
end
mu0 = mu0 / mu0_count;
mu1 = mu1 / mu1_count;


%% compute covariance matrices
cov0 = zeros(2,2);
cov1 = zeros(2,2);
cov0_count = 0;
cov1_count = 0;
for i = 1:L
    x_k = train_patterns(:,i);
    if (train_targets(1,i) == 0)
        cov0_count = cov0_count+1;     
        cov0 = cov0 + (x_k - mu0)*(x_k - mu0)';
    else
        cov1_count = cov1_count+1;
        cov1 = cov1 + (x_k - mu1)*(x_k - mu1)';
    end
end
cov0 = cov0 / cov0_count;
cov1 = cov1 / cov1_count;


%% classify test_patterns
for i = 1:N
    x = test_patterns(:,i);
    prob0 = log((1/det(2*pi*cov0))) -0.5*(x-mu0)'*inv(cov0)*(x-mu0);
    prob1 = log((1/det(2*pi*cov1))) -0.5*(x-mu1)'*inv(cov0)*(x-mu1);
    if (prob0 > prob1)
        test_targets(1,i) = 0;
    else
        test_targets(1,i) = 1;
    end
end
