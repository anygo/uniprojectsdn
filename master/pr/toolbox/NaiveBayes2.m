function test_targets = NaiveBayes2(train_patterns, train_targets, test_patterns, params)

%% get indices of classes
idx0 = find(train_targets == 0);
idx1 = find(train_targets == 1);

%% mean values
mu0 = mean(train_patterns(:,idx0),2);
mu1 = mean(train_patterns(:,idx1),2);

%% covariance matrices
sigma0 = cov(train_patterns(:,idx0)');
sigma1 = cov(train_patterns(:,idx1)');

%% extract diagonals
sigma0 = diag(sigma0);
sigma1 = diag(sigma1);

%% classify test_patterns
N = size(test_patterns,2);
test_targets = zeros(1,N);
n_dims = size(train_patterns,1); % always 2 :-)

for i = 1:N
    x = test_patterns(:,i);
    prob0 = 1;
    prob1 = 1;
    for d = 1:n_dims
        prob0 = prob0 * evalGauss1D(x(d),mu0(d),sigma0(d));
        prob1 = prob1 * evalGauss1D(x(d),mu1(d),sigma1(d));
    end
    if (prob0 > prob1)
        test_targets(1,i) = 0;
    else
        test_targets(1,i) = 1;
    end
end

end



function res = evalGauss1D(x, mu, var)

lhs = 1/(sqrt(var*2*pi));
rhs = -0.5*((x-mu)/sqrt(var))^2;
res = lhs*exp(rhs);

end