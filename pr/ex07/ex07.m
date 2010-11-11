%% cleanup
%close all
clear all
clc

%% params
dim = 2;
n_features = 200;

%% shift around 0-vector
%f = [0 0; 0 1; 2 1; -2 -1; 0 -1]';
f = randn(dim, n_features) + ones(dim,n_features)*5;
f_mean = mean(f,2);
f = f - repmat(f_mean,1,n_features);

scatter(f(1,:),f(2,:),'x');
%axis([-3 3 -3 3]);
hold on;
[U S V] = svd(f);


mu = mean(f,2);
%mu = [0;0];
plot([mu(1) mu(1)+U(1,1)*sqrt(S(1,1))], [mu(2) mu(2)+U(1,2)*sqrt(S(1,1))],'r');
plot([mu(1) mu(1)+U(2,1)*sqrt(S(2,2))], [mu(2) mu(2)+U(2,2)*sqrt(S(2,2))],'r');


for i = 1:size(f,2)
    
    pt = U(:,1)'*f(:,i)*U(:,1);
    plot([f(1,i) pt(1)],[f(2,i) pt(2)]);
    scatter(pt(1),pt(2),'x');
    
    pt = U(:,1)'*f(:,i);
    plot([f(1,i) pt(1)],[f(2,i) 0],'d');
    scatter(pt(1),10,'o','black');
    

end
hold off;