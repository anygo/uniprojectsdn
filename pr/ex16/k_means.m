function [mu, label] = k_means(train_patterns, train_targets, k)

%Reduce the number of data points using the k-means algorithm
%Inputs:
%	train_patterns	- Input means
%	train_targets	- Input targets
%	k				- Number of output data points (clusters)
%
%Outputs
%	mu              - resulting clusters
%	targets			- New targets
%	label			- The labels given for each of the original means
%                     in the range [1,k]

[dim,Np] = size(train_patterns);
dist = zeros(k,Np);
label = zeros(1,Np);

% Initialize the mu's
mu		= randn(dim, k);
mu		= sqrtm(cov(train_patterns',1)) * mu + mean(train_patterns')' * ones(1,k);
old_mu	= zeros(dim,k);

switch k,
case 0
    mu      = [];
    label   = [];
case 1
    mu		= mean(train_patterns')';
    label	= ones(1,Np);
otherwise
    while (sum(sum(abs(mu - old_mu) > 1e-5)) > 0),
      old_mu = mu;
      
      % compute the distances of the train patterns to the means
      for i = 1:k,
         dist(i,:) = ...
      end
      
      % assign the labels
      [m, label] = min(dist);
      
      % update the means
      for i = 1:k,
         mu(:,i) = ...;
      end
    end
end
