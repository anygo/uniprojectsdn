function [ mean_vectors best_k ] = kmeans( data_vectors, k, max_iter )
% kmeans-algorithm 
%   kmeans-algorithm for any-dimensional vectors (at least it SHOULD work :-) )

%% initialize
n_data = size(data_vectors, 2);
dim = size(data_vectors, 1);
mean_vectors = zeros(dim, k);

%% randomize
rand_indices = randperm(n_data);

%% compute mean-vectors
step_size = floor(n_data/k);
for kk = 1:k
    from = (kk-1)*step_size+1;
    to = min((kk)*step_size, n_data);
    cur_indices = rand_indices(1, from:to);
    chosen_vectors = data_vectors(:,cur_indices);
    mean_vectors(:,kk) = mean(chosen_vectors,2)';
end
mean_vectors(isnan(mean_vectors)) = 0;

%% loop
best_k = uint8(zeros(1,n_data));
for i = 1:max_iter
    %% reassign each vector
    mean_vectors_new = zeros(size(mean_vectors));
    distances = zeros(k, n_data);
    for kk = 1:k
        for j = 1:n_data
            distances(kk,j) = norm(double(data_vectors(:,j))-mean_vectors(:,kk));
        end
    end
    
    for j = 1:n_data
        [tmp,idx] = sort(distances(:,j));
        best_k(1,j) = idx(1,1);
    end
    
    for kk = 1:k
        mean_vectors_new(:,kk) = mean(data_vectors(:, best_k == kk),2)';
    end
    
    mean_vectors_new(isnan(mean_vectors_new)) = 0;
    diff = sum(sum(abs(mean_vectors-mean_vectors_new)))/(k*dim);
    fprintf('Iteration %d - diff: %f\n', i, diff);
    if (diff < 0.001)
        fprintf('threshold reached... terminating...\n');
        return;
    end
    
    mean_vectors = mean_vectors_new;
end
    

end

