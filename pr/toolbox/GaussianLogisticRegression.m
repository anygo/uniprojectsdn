function test_targets = GaussianLogisticRegression(train_patterns, train_targets, test_patterns, sameCov)

N = length(train_patterns);


class1 = find(train_targets == 0);
class2 = find(train_targets == 1);

%% mean and cov
mu1 = mean(train_patterns(:, class1)')';
mu2 = mean(train_patterns(:, class2)')';

sigma1 = cov(train_patterns(:, class1)');
sigma2 = cov(train_patterns(:, class2)');

if (sameCov == 1)
    sigma = 0.5 * sigma1 + 0.5 * sigma2;
    sigma1 = sigma;
    sigma2 = sigma;
end

%% prior
p1 = length(class1) / N;
p2 = length(class2) / N;

N_test = length(test_patterns);
test_targets = zeros(1, N_test);


%% invert covs
S1_inv = inv(sigma1);
S2_inv = inv(sigma2);


%% computations
t0 = log(p1/p2);

t1 = log(det((2*pi*sigma2)) ./ det((2*pi*sigma1)));
t2 = (mu2' * S2_inv * mu2) - (mu1' * S1_inv * mu1);

a0 = t0 + 1/2 * (t1 + t2);

%% linear term
at = mu1' * S1_inv - mu2' * S2_inv;

%% quadratic term
A = 1/2 * (S2_inv - S1_inv);

%% loop
for k = 1:N_test
    p0 = 1.0 / (1 + exp(-(test_patterns(:,k)' * A * test_patterns(:,k) + at * test_patterns(:,k) + a0)));
    if (p0 > 0.5)
        test_targets(1,k) = 0;
    else
        test_targets(1,k) = 1;
    end
end

end

