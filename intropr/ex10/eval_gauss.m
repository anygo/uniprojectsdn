function y = eval_gauss(x, mu, sigma)
lhs = 1/(sqrt(2*pi) * sigma);
rhs = (x - mu)^2 / sigma^2;
y = lhs * exp(rhs);
end
