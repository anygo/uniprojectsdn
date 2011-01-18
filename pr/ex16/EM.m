function [test_targets, param_struct] = EM(train_patterns, train_targets, test_patterns, Ngaussians)

% Classify using the expectation-maximization algorithm
% Inputs:
% 	train_patterns	- Train patterns
%	train_targets	- Train targets
%   test_patterns   - Test  patterns
%   Ngaussians      - Number for Gaussians for each class (vector)
%
% Outputs
%	test_targets	- Predicted targets
%   param_struct    - A parameter structure containing the parameters of the Gaussians found

classes  = unique(train_targets);    
Nclasses = length(classes);          %Number of classes in targets
Dim      = size(train_patterns,1);

%The initial guess is based on k-means preprocessing. If it does not converge after
%max_iter iterations, a random guess is used.
disp('Using k-means for initial guess')
for i = 1:Nclasses,
    % todo: initialize mixture weights, mean, and sigma using k_means
end

% Do the EM: Estimate mixture weight, mean, and covariance for each class 
for c = 1:Nclasses,
    train = find(train_targets == classes(c));
    
    if (Ngaussians(c) == 1),
        % If there is only one Gaussian, there is no need to do a whole EM
        % procedure (use Log-Likelihood instead)
        
    else
        
        while ... % convergence criterion %
            
            %E step: Compute Q(theta_i; theta_i+1)
            
            %M step: theta_i+1 <- argmax(Q(theta_i; theta_i+1))
            
            % calculate next estimate for mu (i+1)
            
            % calculate next estimate for sigma (i+1) 
                        
            % calculate weights for the Gaussians
            
        end
    end
end

% classify test patterns

end

% Normal distribution
function p = p_normal(x, mu, sigma)

%Return the probability on a Gaussian probability function. Used by EM
p = ...;

end

