% cleanup
close all;
clear all;
clc;

% Moore-Penrose Pseudoinverse
A = rand(3,3); % just a random square matrix.. use anything you want here!

% pseudoinverse from pinv (matlab-built-in)
A_pinv = pinv(A)

% pseusodinverse from SVD
[U S V] = svd(A);
S(S ~= 0) = 1./S(S ~= 0); % just a lazy way to invert the diagonal elements
S = S';
A_svd = V*S*U'

% pseudoinverse by definition
A_def = inv(A' * A) * A'

% if A is invertible, the equation can be reformulated as
if rank(A) == max(size(A))
    A_def2 = inv(A' * A) * A' % original equation
    A_def2 = inv(A) * inv(A') * A' % "massaging" (at least that is what Prof. H. would say ;) )
    A_def2 = inv(A) * (inv(A') * A') % right hand side: A^-1 * A = Identity!
    A_def2 = inv(A) % i.e. pinv(A) = inv(A)  if A is invertible :-)
end

% All matrices should be the same (A_def2 only if A is invertible :-) )
