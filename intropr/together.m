function Inew = together(I,n_bits, max_iter)
%% 3d (n bit alltogether)
I3dnew = reshape(I,size(I,1)*size(I,2), 3)';
[ cluster_means best_k ] = kmeans(I3dnew, 2^n_bits, max_iter);

best_k = reshape(best_k,size(I,1),size(I,2));

Inew = I;

for a = 1:size(I, 1)
    for b = 1:size(I, 2)
        for i = 1:2^n_bits
            if (best_k(a,b) == i)
                Inew(a,b,:) = cluster_means(:,i);
            end
        end
    end
end
end

