function x = opti4(A,b)
[U S V] = svd(A);
S_inv = zeros(size(S));
S_inv(S~=0) = 1 ./ S(S~=0);
S_inv = S_inv';
A_inv = V*S_inv*U';
x = A_inv*b;
end
