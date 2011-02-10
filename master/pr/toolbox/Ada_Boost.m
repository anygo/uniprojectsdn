function [test_targets, E] = Ada_Boost(train_patterns, train_targets, test_patterns, params)

% Classify using the AdaBoost algorithm
% Inputs:
% 	train_patterns	- Train patterns
%	train_targets	- Train targets
%   test_patterns   - Test  patterns
%	Params	- [NumberOfIterations, Weak Learner Type, Learner's parameters]
%
% Outputs
%	test_targets	- Predicted targets
%   E               - Errors through the iterations
%
% NOTE: Suitable for only two classes
%

[maxIter, weak_learner, alg_param] = process_params(params);
%[Ni,M]			= size(train_patterns);
[d,nTrain]		= size(train_patterns);
% Weiht vector for each training sample
w			 	= ones(1,nTrain)/nTrain;
IterDisp		= 10;

full_patterns   = [train_patterns, test_patterns];
test_targets    = zeros(1, size(test_patterns,2));

%Do the AdaBoosting
for k = 1 : maxIter,
   % Train weak learner Ck using the data sampled according to W:
   % ...so sample the data according to W
   randnum = rand(1,nTrain);	% return n numbers between 0 and 1
   cW	   = cumsum(w);		 	% return cumulative sum: (1,3,5) = (1,4,9)
   indices = zeros(1,nTrain); 	
   
   for i = 1:nTrain,
      % Find which bin the random number falls into
	  % Take the current random number
	  % Find all training samples whose weight is smaller than that number
	  % Find the element that whose cumulative weight is closest to the rand number
	  % As the weight is larger for those elements, they cover a greater part of the
	  % line [0,1], and so are chosen more probable than other test patterns!
      loc = find(randnum(i) > cW, 1, 'last' ) + 1;
      if isempty(loc)
         indices(i) = 1;
      else
         indices(i) = loc;
      end
   end
   %indices

   %...and now train the classifier with the patterns that were chosen, i.e., not classified correctly previously
   Ck 	= feval(weak_learner, train_patterns(:, indices), train_targets(indices), full_patterns, alg_param);

   % Ek <- Training error of Ck 
   E(k) = sum(w.*(Ck(1:nTrain) ~= train_targets));
   
   if (E(k) == 0)
      break
   end
   
   % Classifier weight
   %alpha_k <- 1/2*ln(1-Ek)/Ek)
   alpha_k = 0.5*log((1-E(k))/E(k));
   
   % Update the train_pattern weights
   %W_k+1 = W_k/Z*exp(+/-alpha)
   w  = w.*exp(alpha_k*(xor(Ck(1:nTrain),train_targets)*2-1));
   w  = w./sum(w);
   
   %Update the test targets
   test_targets  = test_targets + alpha_k*(2*Ck(nTrain+1:end)-1);
   
   % For every 10th boosting step, print message
   if (k/IterDisp == floor(k/IterDisp)),
      disp(['Completed ' num2str(k) ' boosting iterations'])
   end
   
end

test_targets = test_targets > 0;