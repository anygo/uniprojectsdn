function B = opti1(A)
[U S V] = svd(A);
%s(1,:) = S(S~=0);
minimum_dim = min(size(A));
S(minimum_dim:end, minimum_dim:end) = 0;
B = U*S*V';

subplot(1,3,1);
imagesc(A);
subplot(1,3,2);
imagesc(B);
subplot(1,3,3);
imagesc(A-B);
end