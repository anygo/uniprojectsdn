function test_targets = kNearestNeighbors(train_patterns, train_targets, test_patterns, params)

% Classify using the Nearest neighbor algorithm
% Inputs:
% 	train_patterns	- Train patterns
%	train_targets	- Train targets
%   test_patterns   - Test  patterns
%	params		    - unused
%
% Outputs
%	test_targets	- Predicted targets

L			= length(train_targets);
Uc          = unique(train_targets);
k           = params;

N               = size(test_patterns, 2);
test_targets    = zeros(1,N); 
for i = 1:N,
    dist            = sum((train_patterns - test_patterns(:,i)*ones(1,L)).^2);
    [m, indices]    = sort(dist);
    
    best            = train_targets(indices(1:k));

    [bins class]    = hist(best, Uc);
    
    [bla idx]       = max(bins);
    
    test_targets(i) = class(idx);
end
