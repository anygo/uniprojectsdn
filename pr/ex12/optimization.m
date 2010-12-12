function [] = optimization()
    
    maxIter = 10000;
    epsilon = 1e-5;
    usedNorm = 2;

    % init
    x = [0.9; 0.6];
    
    % matlab steepest descent
    %[x,fval] = fminunc(@wrapper, x);
    
    plot_function(x);
    
    % path of steps (just for visualization)
    opt_path = [x; wrapper(x)];
    
    for i = 1:maxIter
        % compute highest descent direction
        grad_f_x = [f(x(1), x(2)) - f(x(1)-epsilon, x(2)); ...
            f(x(1), x(2)) - f(x(1), x(2)-epsilon)];
        
        if usedNorm == 1
            if abs(grad_f_x(1)) > abs(grad_f_x(2))
                delta_x = [-sign(grad_f_x(1)); 0];
            else
                delta_x = [0; -sign(grad_f_x(2))];
            end
        elseif usedNorm == 2
            delta_x = -grad_f_x/norm(grad_f_x, 2);
        elseif usedNorm == 'P'
            
        end
        
        % line search
        t = armijo_goldstein(x, delta_x, grad_f_x);
        
        % update
        x_prev = x;
        x = x + t*delta_x;
        
        % add to path
        new_path = [x; wrapper(x)];
        opt_path = [opt_path new_path];
        
        % convergence check
        if abs(x_prev - x) < epsilon
            break
        end
        
        plot_function(x);
        drawnow;
    end
    
    disp(['converged after ' num2str(i) ' steps (L' num2str(usedNorm) ' norm) --> ' ...
        'f(' num2str(x(1)) ', ' num2str(x(2)) ') = ' num2str(wrapper(x))])
    
    % plot
    plot_function(x);
    
    % plot path
    subplot(1,2,1);
    hold on;
    plot3(opt_path(1,:), opt_path(2,:), opt_path(3,:));
    hold off;
    subplot(1,2,2);
    hold on;
    plot(opt_path(1,:), opt_path(2,:));
    hold off;

end

% line search (1D optimization)
function t = armijo_goldstein(x, delta_x, grad_f_x)
    alpha = 0.25; % alpha in [0, 0.5]
    beta = 0.75; % beta in [0, 1]
    t = 1; 
    while wrapper(x + t*delta_x) > wrapper(x) + alpha*t*grad_f_x'*delta_x;
        t = beta*t;
    end
end

function plot_function(x)
    [X1, X2] = meshgrid(-2:0.05:1, -1:0.05:1);
    
    y = f(X1(:)', X2(:)');
    y = reshape(y, size(X1, 1), size(X2, 2));
    
    figure(1)
    
    subplot(1,2,1)
    surfc(X1, X2, y, 'FaceColor','green','EdgeColor','none');
    camlight left; lighting phong
    set(gca, 'GridLineStyle', '--');
    xlabel('x_1');
    ylabel('x_2');
    zlabel('f(x_1, x_2)');
    title('Exponential Function');

    % current point
    hold on;
    scatter3(x(1), x(2), wrapper(x), '+r');
    hold off;
    
   
    subplot(1,2,2)
    [C, h] = contour(X1, X2, y, [3 5 7]);
    set(h,'ShowText','on','TextStep',get(h,'LevelStep'))
    colormap default
    axis equal
    xlabel('x_1');
    ylabel('x_2');
    title('Contour Plot');
    
    % current point
    hold on;
    scatter(x(1), x(2), '+r');
    hold off;
end

% wrapper function for f(x1, x2), if you want to use a vector of x-values
function val = wrapper(x_vec)
    val = f(x_vec(1), x_vec(2));
end

function val = f(x1, x2)
    val = exp(x1 + 3.*x2 - 0.1) + exp(x1 - 3.*x2 - 0.1) + exp(-x1 - 0.1);
end
