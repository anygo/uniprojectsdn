function optimization_tests()

    cur1 = -4;
    cur2 = 8;
    for i = 1:100
        dx1 = f(cur1)-f(cur1-0.1);
        dx2 = f(cur2)-f(cur2-0.1);
        cur1 = cur1 - dx1;
        cur2 = cur2 - dx2;
        
        x2 = cur2-0.5:0.01:cur2+0.5;
        y2 = zeros(size(x2,2));
        for j = 1:size(x2,2)
            y2(j) = f(x2(j));
        end
        x1 = cur1-0.5:0.01:cur1+0.5;
        y1 = zeros(size(x1,2));
        for j = 1:size(x1,2)
            y1(j) = f(x1(j));
        end
        plot(x1,y1);
        hold on;
        plot(x2,y2);
        scatter(cur1, f(cur1), 'r');
        scatter(cur2, f(cur2), 'r');
        hold off;
        axis([-5 9 0 10]);
        drawnow;
        if abs(cur1-cur2) < 0.01
            break;
        end
    end
end

function y = f(x)
    if x < 1
        y = -2*x + 4;
    end
    if x > 2.5
        y = x - 1.25;
    end
    if (x >= 1 && x <= 2.5)
        y = (x-2)^2+1; 
    end
end