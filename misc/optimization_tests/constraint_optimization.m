function constraint_optimization()
    x = -5:0.1:5;
    
    plot(x, f(x));
    hold on;
    %plot(x, constrained(x), 'r');

    %plot(-5:0.1:5,g(-5:0.1:5));
    g_lambda = zeros(1,101);
    for i = 1:size(g_lambda,2)
        g_lambda(i) = g(x(i)); 
    end
    plot(-5:0.1:5, g_lambda, 'r');
    
    lam = fminunc(@g, 0);
    global lambda_global;
    lambda_global = lam;
    plot(x, constrained(x), 'g');
    hold off;
end

function y = g(lambda)
    global lambda_global;
    lambda_global = lambda;

    x = fminunc(@constrained, 0);
    y = constrained(x);
    y = -y;
end

function y = f(x)
    y = 0.1*x.^4 - 2*x.^2 + x;
end

function y = constrained(x)
    global lambda_global;
    y = f(x) + lambda_global*(-x-2); % constraint: x >= 2  ==  -x <= 2
end