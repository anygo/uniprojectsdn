% EVALREGRESSIONTREE1D - Calculate regression tree outputs for a given
%                        vector of measurements.
%
% y = evalRegressionTree1D(tree, x) calculates the outputs Y of the 
% regression tree TREE using the N 1-dimensional measurment vector X.
%
% y = evalRegressionTree1D(..., hFigure) evaluates and plots the tree
% regions to figure with handle HFIGURE.
function y = evalRegressionTree1D(tree, x, hFigure)

    numX = length(x);
    y = zeros(1, numX);
    
    % For each value in the given vector x...
    for k = 1:numX
        y(k) = evalSingleMeasurment(tree, x(k));
    end
    
    if nargin > 2
        % Plot the tree regions if required.
        figure(hFigure);
        bar(x, y, 1);
        grid on;
        axis([min(x) max(x) min(y) 1.1 * max(y)]);
    end
    

function y = evalSingleMeasurment(tree, x)

    % Traverse through the regression tree.
    subTree = tree;
    while ( ~isempty(subTree.left) && ~isempty(subTree.right) )
        if x < subTree.splitValue
            % Goto the left subtree.
            subTree = subTree.left;
        else
            % Goto the right subtree.
            subTree = subTree.right;
        end
    end
    
    % The end of the tree structure is reached.
    y = subTree.approxValue;
        