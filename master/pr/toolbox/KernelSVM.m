function test_targets = KernelSVM(train_patterns, train_targets, test_patterns, param)

[Dim, Np]       = size(train_patterns);

% extend dimension
% Dim             = Dim + 1;
% train_patterns(Dim,:) = ones(1,Np);
% test_patterns(Dim,:)  = ones(1, size(test_patterns,2));

% get [-1,1] class info
yi = train_targets;
yi(find(yi == 0)) = -1;

% kernel width
width = param;


% create the kernel matrix from the input patterns using Gaussian RBF
K = zeros(Np); 
for i = 1:Np
    K(:,i) = GaussianRBF(train_patterns(:,i), train_patterns, width);
end

%in the case of slack variables:  0 <= alpha_i <= slack
slack = 0.1; %mu

% inequality constraint that lambda_i >= 0
A = -eye(Np);
b = zeros(Np, 1);

% equality constraint that sum(yi*lambda_i) = 0 
Aeq = yi;
beq = 0;

% Quadratic programming
options = optimset('LargeScale', 'off');
H = diag(yi) * K * diag(yi);
f = - ones(1, Np);

% alpha_star = solution to the dual problem!
alpha_star	= quadprog(H, f, A, b, Aeq, beq, zeros(1, Np), slack*ones(1, Np), [], options)';

% a_star is the solution to the primal problem! sum(alpha_i*yi*xi)
a_star		= (alpha_star.*yi) * K';


%Find the bias
sv_for_bias  = find((alpha_star > 0.001*slack) & (alpha_star < slack - 0.001*slack));
if isempty(sv_for_bias) % no support vectors
   bias     = 0;
else
   B        = yi(sv_for_bias) - a_star(sv_for_bias); %average bias
   bias     = mean(B);
end

sv = find(alpha_star > 0.001*slack);
    
% classify test targets
N = length(test_patterns);
y = zeros(1,N);
for i = 1:length(sv)
    v = a_star(sv(i)) * GaussianRBF(train_patterns(:,sv(i)), test_patterns, width);
    y = y + v';
end

test_targets = y + bias;
test_targets = test_targets > 0;

return

end

% Kernel function
function k = GaussianRBF(x, patterns, width)
    k = 1./ (sqrt(2*pi)*width) .* exp( - sum((patterns - x * ones(1, length(patterns))).^2 )' ./ (2 * width^2) );
end
