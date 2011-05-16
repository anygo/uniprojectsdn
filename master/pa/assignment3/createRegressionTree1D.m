% CREATEREGRESSIONTREE1D - Create a 1-dimensional regression tree.
%
function tree = createRegressionTree1D(x, y)

    if length(x) > 1   
        % Calculate an optimal split position.
        splitPos = findSplitPos(x, y);      
        
        tree.splitValue = x(splitPos);
        tree.approxValue = [];
        % Create left and right subtree.
        tree.left = createRegressionTree1D( x(1:(splitPos-1)), y(1:(splitPos-1)) );
        tree.right = createRegressionTree1D( x(splitPos:end), y(splitPos:end) );
        return;
       
    end           
   
    % Create a terminal node.
    tree.splitValue = x;          % The value which is used for splitting.
    tree.approxValue = mean(y);   % The approximation used in the region.
    tree.left = [];               % The reference to the left subtree.
    tree.right = [];              % The reference to the right subtee.
          
function [splitPos, minErr] = findSplitPos(x, y)

    numX = length(x);
    minErr = Inf;
    splitPos = 1;
    
    % Iterate over all possible split positions.
    for k = 2:numX     
        % Split and calculate the error.
        errLeft = getError( y(1:k-1) );
        errRight = getError( y(k:end) );
        errTotal = errLeft + errRight;
        if (errTotal < minErr)
            % Found new optimal split position.
            minErr = errTotal;
            splitPos = k;
        end     
    end
    
function err = getError(y)

    err = length(y) * var(y);