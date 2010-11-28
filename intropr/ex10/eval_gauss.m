function y = eval_gauss(x, mu, sigma)

lhs = 1/(sqrt(2*pi) * sigma);
x = x';
y = zeros(size(x));
for i = 1:size(x,2)
    rhs = -0.5 * (x(:,i) - mu)^2 / sigma^2;
    y(:,i) = lhs * exp(rhs);
end
end
