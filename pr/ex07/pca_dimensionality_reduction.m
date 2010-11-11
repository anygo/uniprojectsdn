function reduced = pca_dimensionality_reduction(orig)

dim = size(orig,1);
N = size(orig,2);

orig_mean = mean(orig,2);
orig = orig - repmat(orig_mean,1,N);

% initialize dimensionality reduced return matrix
reduced = zeros(dim-1,N);

[U S V] = svd(orig);


for i = 1:N
    reduced(:,i) = U(:,1:dim-1)'*orig(:,i);
end

transformed_mean = U(:,1:dim-1)'*orig_mean;

reduced = reduced + repmat(transformed_mean,1,N);

end