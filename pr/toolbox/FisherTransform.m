function [patterns, train_targets, phi] = FisherTransform(train_patterns, train_targets, param, plot_on)
%Reshape the data points using the Fisher Transform
%Inputs:
%	train_patterns	- Input patterns
%	train_targets	- Input targets
%	param			- Unused
%   plot_on         - Unused
%
%Outputs
%	patterns		- New patterns
%	targets			- New targets
%   phi				- transformation matrix

% Task: 
% Implement a transform that maximizes the inter-class and minimized the 
% intra-class distance. The transform is then returned as an output
% argument, together with the transformed patterns and the corresponding
% train_targets.

Uclasses = unique(train_targets);

% get number of classes and number of features
Nc = length(Uclasses);
[Dim,Np] = size(train_patterns);

% initial transformation
phi = eye(Dim, Dim);


% compute interclass kernel matrix E_inter

% Hint: This is the same matrix as used in the LDA classification!
% you will need the mean of all class means

% compute intraclass kernel matrix E_intra

% create the combined kernel matrix E

% solve the eigenvalue / eigenvector problem for E

% create phi such that it maximizes the Rayleigh ratio
% phi = ...

% apply feature transform
patterns = phi * train_patterns;

if Dim > 2
    % reduce dimensions of transformed features
    patterns = [patterns(1,:); patterns(2,:)];
end

