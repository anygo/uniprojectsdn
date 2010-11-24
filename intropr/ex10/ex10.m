%% load image
I = imread('object.png');

[counts x] = imhist(I,128);
%counts = counts/sum(counts);

thresh = x(1);
thresh = 20000;


vec = [];
for i = 65536/thresh:size(x,1)
   for j = 1:counts(i)
      vec = [vec; x(i)]; 
   end
end

phat = mle(vec);