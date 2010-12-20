function test_targets = SVM(train_patterns, train_targets, test_patterns, softmargin)

[Dim,Np] = size(train_patterns);

%change classes 0 to -1
train_targets(find(train_targets == 0)) = -1;

if softmargin == 1
    % we need access to these variables during the optimization
    assignin('base', 'selected_targets', train_targets);
    assignin('base', 'selected_patterns', train_patterns);
end

%initialize
a = [0, 1, 0];

% options for fmincon
options = optimset('GradObj','on','MaxFunEvals',1e6, 'MaxIter', 100);

if softmargin == 0
    % linear inequalities
    %create matrix A, vector b
    A = zeros(Np, Dim + 1);
    b = ones(Np,1);

    for i = 1:Np
        A(i,1) = (train_patterns(1,i)' .* train_targets(i));
        A(i,2) = (train_patterns(2,i)' .* train_targets(i));
        A(i,3) = (train_targets(i));
    end
    
    % the fmincon function requires: c <= 0 !
    A = -A;
    b = -b;

    % Start
    [vec,fval,exitflag] = fmincon(@anorm,a,A,b,[],[],[],[],[],options);
    
    % value for residual
    sum(A*vec' - b)
else
    
    % in case of the softmargin, we have nonlinear inequalities and need to
    % provide a dedicated function to evaluate the constraint at each
    % iteration

    % initialize
    x0 = [0 1 0 zeros(1,size(train_patterns,2))];
    [vec,fval,exitflag] = fmincon(@anormslack,x0,[],[],[],[],[],[],@slackCon,options);
end

% result parameters
vec

% Classify test patterns
test_targets = zeros(1, length(test_patterns));
v=zeros(2,1);
v(1) = vec(1);
v(2) = vec(2);

for i = 1:length(test_patterns)
    test_targets(i) = (v' * test_patterns(:,i) + vec(3)) > 0;    
end

end

% Objective function for hard margins
function [res, g] = anorm(a)
    v=zeros(2,1);
    v(1) = a(1);
    v(2) = a(2);
    res = 1/2 * norm(v)^2;
    g = [a(1), a(2), 0];
end

% Objective function for soft margins with slack variables
function [res, g] = anormslack(a)
    mu = 1;

    v=zeros(2,1);
    v(1) = a(1);
    v(2) = a(2);
    res = 1/2 * norm(v)^2 + mu*sum(a(4:end));
    g = [a(1), a(2), 0, ones(1,size(a,2)-3)*mu];
end

% Nonlinear constraints for soft margin problem with slack variables
% Slack constraints
function [c,cleq] = slackCon(a)
    train_targets = evalin('base', 'selected_targets');
    train_patterns = evalin('base', 'selected_patterns');
    
    c = zeros(size(train_patterns,2)*2,1);
    for i=1:size(train_patterns,2)
        x = train_patterns(:,i);
        y = train_targets(i);
        c(i) = -(y*(a(1:2)*x + a(3)) -1 + a(i+3));
    end
    c(size(train_patterns,2)+1:end) = -a(4:end);
    cleq = [];
end