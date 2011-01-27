function [test_targets, param_struct] = EMClassifier(train_patterns, train_targets, test_patterns, Ngaussians)

    % Classify using the expectation-maximization algorithm
    % Inputs:
    % 	train_patterns	- Train patterns
    %	train_targets	- Train targets
    %   test_patterns   - Test  patterns
    %   Ngaussians      - Number for Gaussians for each class (vector)
    %
    % Outputs
    %	test_targets	- Predicted targets
    %   param_struct    - A parameter structure containing the parameters of the Gaussians found

    classes             = unique(train_targets);    %Which classes do we have (0,1)
    Nclasses            = numel(classes);          %Number of classes in targets (2)
    Dim                 = size(train_patterns,1);   %Dimensionality of features

    max_iter   			= 100;
    max_try             = 5;
    Pw					= zeros(Nclasses,max(Ngaussians));%contains priors for the Gaussians
    sigma				= zeros(Nclasses,max(Ngaussians),size(train_patterns,1),size(train_patterns,1));%sigmas for all Gaussians
    mu					= zeros(Nclasses,max(Ngaussians),size(train_patterns,1));%mu's for all gaussians

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %The initial guess is based on k-means preprocessing. 
    disp('Using k-means for initial guess')
    for i = 1:Nclasses, %initialize Gaussian parameters for each class separately
        in  = find(train_targets==classes(i));
        [initial_mu, labels]	= k_means_em(train_patterns(:,in),Ngaussians(i));
        %now we have the mu's for the gaussians, and a vector labels that says
        %to which Gaussian (1,2,3,....) the individual train_patterns belong to

        for j = 1:Ngaussians(i), %prior and covariance matrix for each gaussian
            gauss_labels    = find(labels == j); %find features of gaussian j
            Pw(i,j)         = length(gauss_labels) / length(labels);% #feat. in gaussian j / all class features
            %cmp exercise: as initialization, use square of std.dev. on diagonal!
            sigma(i,j,:,:)  = diag(std(train_patterns(:,in(gauss_labels))').^2);
        end

        mu(i,1:Ngaussians(i),:) = initial_mu'; %we got that from the k-means already
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %Do the EM: Estimate mean and covariance for each class separately
    for c = 1:Nclasses,
        train	= find(train_targets == classes(c));

        if (Ngaussians(c) == 1),
            %If there is only one Gaussian, there is no need to do a whole EM procedure
            sigma(c,1,:,:)  = cov(train_patterns(:,train)',1);
            mu(c,1,:)       = mean(train_patterns(:,train)');
        else

            sigma_l         = squeeze(sigma(c,:,:,:));      %just get the sigmas for the current class
            old_sigma       = zeros(size(sigma_l)); 		%Used for the stopping criterion
            iter			= 0;							%Iteration counter
            n			 	= length(train);				%Number of training points
            qi			    = zeros(Ngaussians(c),n);       %qi = (p_l * N(c_j,B_l) / sum_k(p_k * N(c_j,B_k)))
            P				= zeros(1,Ngaussians(c));       
            Ntry            = 0;

            %check whether the covariance matrix still changes significantly
            while ((sum(sum(sum(abs(sigma_l-old_sigma)))) > 1e-4) & (Ntry < max_try))

                old_sigma = sigma_l;

                %E step: Compute Q(theta; theta_i)

                for j = 1:n,
                    % for all features of my class, get a single one
                    data  = train_patterns(:,train(j));
                    for k = 1:Ngaussians(c),
                        %this is the numerator for the qi, i.e. the probability
                        %of a feature belonging to a certain gaussian,
                        %multiplied with the prior for this gaussian
                        P(k) = Pw(c,k) * p_single(data, squeeze(mu(c,k,:)), squeeze(sigma_l(k,:,:)));
                    end          

                    for l = 1:Ngaussians(c),
                        %cmp initialization of qi above. it contains the
                        %probability of a feature c_j to belong to a certain
                        %gaussian, normalized by the sum of the probabilites
                        %for all class-gaussians, so that they add up to one
                        qi(l,j) = P(l) / sum(P); 
                    end
                end

                % Finally calculate the priors for the gaussians
                Pw(c,1:Ngaussians(c)) = 1/n * sum(qi');

                %M step: theta_i+1 <- argmax(Q(theta; theta_i))

                %Calculating mu's
                for l = 1:Ngaussians(c),
                    %expand the qi so that we can apply an elementwise
                    %multiplication. 
                    mu(c,l,:) = sum((train_patterns(:,train).*(ones(Dim,1)*qi(l,:)))')/sum(qi(l,:)');
                end

                %Calculating sigma's
                %A bit different from the handouts, but much more efficient
                for l = 1:Ngaussians(c),
                    data_vec = train_patterns(:,train); %class features
                    %generates term (c_j - mu_l) already for all features
                    data_vec = data_vec - squeeze(mu(c,l,:)) * ones(1,n);
                    %the numerator can be drawn into the term (c_j -mu_l), but
                    %since we will multiply it with itself to get the
                    %covariance, I have to take the square root to make it
                    %right!
                    data_vec = data_vec .* (ones(Dim,1) * sqrt(qi(l,:)));
                    %all that remains is the covariance ()*()^T
                    % the cov() function normalizes by 1/n, since we do not want that, we have to multiply by n 
                    sigma_l(l,:,:) = cov(data_vec',1)*n/sum(qi(l,:)');
                end

                iter = iter + 1;
                disp(['Iteration: ' num2str(iter)])

                %checking convergence...
                if (iter > max_iter),
                    theta = randn(size(sigma_l));
                    iter  = 0;
                    Ntry  = Ntry + 1;

                    if (Ntry > max_try)
                        disp(['Could not converge after ' num2str(Ntry-2) ' redraws. Quitting']);
                    else
                        disp('Redrawing weights.')
                    end
                end

            end

            sigma(c,:,:,:) = sigma_l;
        end
    end

    %Classify test patterns
    %Build struct for classes with respective parameters for the gaussians
    for c = 1:Nclasses,
        param_struct(c).p       = length(find(train_targets == classes(c)))/length(train_targets);
        param_struct(c).mu      = squeeze(mu(c,1:Ngaussians(c),:));
        param_struct(c).sigma   = squeeze(sigma(c,1:Ngaussians(c),:,:));
        param_struct(c).w       = Pw(c,1:Ngaussians(c));
        for j = 1:Ngaussians(c)
            param_struct(c).type(j,:) = cellstr('Gaussian');
        end
        if (Ngaussians(c) == 1)
            param_struct(c).mu = param_struct(c).mu';
        end
    end
    %classification
    test_targets = classify_parametric(param_struct, test_patterns);

end

function p = p_single(x, mu, sigma)

    if length(mu) == 0
        p = 0;
    else
        %Return the probability on a Gaussian probability function. Used by EM
        p = 1/sqrt(det(2*pi*sigma)) *  exp(-0.5 * (x-mu)' * inv(sigma) * (x-mu));
    end
end

% merged k_means and EM
function [mu, label] = k_means_em(train_patterns, k)

    %Reduce the number of data points using the k-means algorithm
    %Inputs:
    %	train_patterns	- Input means
    %	train_targets	- Input targets
    %	k				- Number of output data points (clusters)
    %
    %Outputs
    %	mu              - resulting clusters
    %	targets			- New targets
    %	label			- The labels given for each of the original means
    %                     in the range [1,k]

    [dim,Np] = size(train_patterns);
    dist = zeros(k,Np);
    label = zeros(1,Np);

    % Initialize the mu's
    mu		= sqrtm(cov(train_patterns',1)) * randn(dim, k) + mean(train_patterns')' * ones(1,k);
    old_mu	= zeros(dim,k);

    switch k,
    case 0
        mu      = [];
        label   = [];
    case 1
        mu		= mean(train_patterns')';
        label	= ones(1,Np);
    otherwise
        while (sum(sum(abs(mu - old_mu) > 1e-5)) > 0),
          old_mu = mu;

          % compute the distances of the train patterns to the means
          for i = 1:k
             dist(i,:) = sum((train_patterns - repmat(mu(:,i), 1, size(train_patterns, 2))).^2)';
          end

          % assign the labels
          [m, label] = min(dist);

          % update the means
          for i = 1:k,
             mu(:,i) = mean(train_patterns(:,label == i),2);
          end
        end
    end
end
