function [cdf normalized_hist] = compute_cdf(img, bins)
[counts x] = imhist(img, bins);
normalized_hist = counts / sum(counts);

cdf = zeros(bins,1);
cdf(1,1) = normalized_hist(1);
for i = 2:bins
   cdf(i) = cdf(i-1) + normalized_hist(i); 
end

end

