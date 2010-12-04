function test_targets = RosenbergsPerceptron(train_patterns, train_targets, test_patterns, max_iter)
    %% rerepresent classes for Rosenbergs Perceptron {0; 1} -> {-1; 1}
    train_targets(train_targets == 0) = -1;

    %% minimization
    [alpha_0, alpha] = minimize(train_patterns, train_targets, max_iter);
    
    %% compute test_targets
    test_targets = zeros(1,size(test_patterns,2));
    for i = 1:size(test_patterns,2)
        x = test_patterns(:,i);
        test_targets(1,i) = sign(alpha' * x + alpha_0);
    end

    %% rererepresent classes for the classifier-toolbox {-1; 1} -> {0; 1}  
    test_targets(test_targets == -1) = 0;
end

function [alpha_0 alpha] = minimize(x, y, max_iter)
    alpha_0 = 0;
    alpha = [0; 0];
    
    for i = 1:max_iter     
        % choose one misclassified pattern
        cur_idx = -1;
        for j = 1:size(x,2)
            if y(j) * (x(:,j)' * alpha + alpha_0) <= 0
                 cur_idx = j;
            end
            if cur_idx ~= -1
                break;
            end
        end
        if cur_idx == -1
            % no missclassified patterns left
            return;
        end
                
        % update parameters
        alpha_0 = alpha_0 + y(1,cur_idx);
        alpha = alpha + y(1,cur_idx)*x(:,cur_idx);     
    end
    
    if i == max_iter
        disp('!!! warning: max_iter reached...');
    end
end



% function res = obj_func(alpha_0, alpha, M, M_targets)
%     res = 0;
%     for i = 1:size(M,2)
%         x_i = M(1,i);
%         res = res + M_targets(1,i) * (alpha' * x_i - alpha_0);
%     end
%     res = -res;
% end
% 
% function [d_alpha_0 d_alpha]= compute_gradient(M, M_targets)
%     d_alpha_0 = - sum(M_targets);
%     d_alpha   = - sum(M_targets .* M);
% end
