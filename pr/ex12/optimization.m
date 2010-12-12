function [] = optimization()

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
   
    subplot(1,2,2)
    [C, h] = contour(X1, X2, y, [1 2 3 4 5 6 7 8 9 10]);
    set(h,'ShowText','on','TextStep',get(h,'LevelStep'))
    colormap hot
    axis equal
    xlabel('x_1');
    ylabel('x_2');
    title('Contour Plot');

end


function val = f(x1, x2)
    val = exp(x1 + 3.*x2 - 0.1) + exp(x1 - 3.*x2 - 0.1) + exp(-x1 - 0.1);
end
