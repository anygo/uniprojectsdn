function Inew = separately(I, n_bits, max_iter)
%% each channel separately (n bit per channel)
Ir = I(:,:,1);
Ig = I(:,:,2);
Ib = I(:,:,3);
data_vectorsR = reshape(Ir, 1, size(Ir,1)*size(Ir,2));
[ cluster_meansR best_kR ] = kmeans(data_vectorsR, 2^n_bits, max_iter); 
data_vectorsG = reshape(Ig, 1, size(Ig,1)*size(Ig,2));
[ cluster_meansG best_kG ] = kmeans(data_vectorsG, 2^n_bits, max_iter); 
data_vectorsB = reshape(Ib, 1, size(Ib,1)*size(Ib,2));
[ cluster_meansB best_kB ] = kmeans(data_vectorsB, 2^n_bits, max_iter); 

Inew(:,:,1) = reshape(best_kR, size(Ir,1), size(Ir,2));
Inew(:,:,2) = reshape(best_kG, size(Ig,1), size(Ig,2));
Inew(:,:,3) = reshape(best_kB, size(Ib,1), size(Ib,2));


for a = 1:size(Inew, 1)
    for b = 1:size(Inew, 2)
        for i = 1:2^n_bits
            if (Inew(a,b,1) == i) 
                Inew(a,b,1) = cluster_meansR(:,i);
            end
            if (Inew(a,b,2) == i) 
                Inew(a,b,2) = cluster_meansG(:,i);
            end
            if (Inew(a,b,3) == i) 
                Inew(a,b,3) = cluster_meansB(:,i);
            end
        end
    end
end
end

