function test_targets = SVM(train_patterns, train_targets, test_patterns, params)
    %% rerepresent classes for Rosenblatts Perceptron {0; 1} -> {-1; 1}
    train_targets(train_targets == 0) = -1;

    %% create linear inequations  Ax <= b
    % x = [alpha_0; alpha(0); alpha(1)]
    A = zeros(size(train_patterns, 2), 3);
    b = -ones(size(train_patterns, 2), 1);
    for i = 1:size(train_patterns, 2)
        A(i, 1) = -train_targets(1, i);
        A(i, 2) = -train_targets(1, i) * train_patterns(1, i);
        A(i, 3) = -train_targets(1, i) * train_patterns(2, i);
    end

    %% find minimum
    x0 = [0; 0; 1];
    alpha_vec = fmincon(@obj_func, x0, A, b);
    
    alpha_0 = alpha_vec(1);
    alpha = alpha_vec(2:3);
    
    %% assign classes
    test_targets = zeros(1,size(test_patterns,2));
    for i = 1:size(test_patterns,2)
        x = test_patterns(:,i);
        test_targets(1,i) = sign(alpha' * x + alpha_0);
    end
    
    %% rererepresent classes for the classifier-toolbox {-1; 1} -> {0; 1}  
    test_targets(test_targets == -1) = 0;
end

% alpha is [alpha_0; alpha(0); alpha(1)] (siehe Übungsblatt (selbst
% geschriebenes zeug))
function val = obj_func(alpha)
    val = norm(alpha(2:3), 2)^2 / 2;
end