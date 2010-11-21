function [patterns, train_targets, phi] = FisherTransform(train_patterns, train_targets, param, plot_on)

% Task: 
% Implement a transform that maximizes the inter-class and minimized the 
% intra-class distance. The transform is then returned as an output
% argument, together with the transformed patterns and the corresponding
% train_targets.

Uclasses = unique(train_targets);

[Dim,N] = size(train_patterns);

% initial transformation
phi = eye(Dim, Dim);


% compute interclass kernel matrix 
mu_joint = mean(train_patterns,2);
E_inter = zeros(Dim, Dim);
for i = 1:N
    x = train_patterns(:,i);
    E_inter = E_inter + (x-mu_joint)*(x-mu_joint)';
end
E_inter = E_inter / N;

% Hint: This is the same matrix as used in the LDA classification!
% you will need the mean of all class means
% compute intraclass kernel matrix E_intra
class0 = train_targets == 0;
class1 = train_targets == 1;

mu0 = mean(train_patterns(:, class0),2);
mu1 = mean(train_patterns(:, class1),2);

E_intra = zeros(Dim, Dim);
for i = 1:N
    x = train_patterns(:,i);
    if train_targets(1,i) == 0
        E_intra = E_intra + (x-mu0)*(x-mu0)';
    else
        E_intra = E_intra + (x-mu1)*(x-mu1)';
    end
end

% create the combined kernel matrix E
E = inv(E_intra)*E_inter;

% solve the eigenvalue / eigenvector problem for E
[U S V] = svd(E);
a = U(1,:);

% create phi such that it maximizes the Rayleigh ratio
phi = eye(Dim,Dim)*a;

% apply feature transform
patterns = phi * train_patterns;

if Dim > 2
    % reduce dimensions of transformed features
    patterns = [patterns(1,:); patterns(2,:)];
end

