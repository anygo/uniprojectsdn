close all;
clear all;

A = [11 10 14; 12 11 -13; 14 13 -66 ];

% determinant
fprintf('determinant = %d\n', det(A));

% compute inverse of A without 'inv()'
[U S V] = svd(A);
Sinv = S;
Sinv(Sinv ~= 0) = 1 ./ Sinv(Sinv ~= 0);
Ainv = V*Sinv*U';
check = Ainv - inv(A);
check(abs(check) < 0.001) = 0;
if check ~= zeros(3,3)
    fprintf('inversion falsch\n');
else
    fprintf('inversion ueber SVD stimmt mit inv() ueberein\n');
end

% condition number
fprintf('condition number = %d (%d)\n', max(max(S)) / min(min(S(S>0))), cond(A))


% ex 1.1
b_orig = [1.001;0.999;1.001];
x_orig = Ainv*b_orig;
b_variation = [1;1;1];
x_variation = Ainv*b_variation;

delta_x = Ainv*(b_variation-b_orig);

