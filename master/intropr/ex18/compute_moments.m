function M_pq = compute_moments(img, p, q, central)

M_pq = 0;
[Y X] = find(img > 0);
mean_X = mean(X);
mean_Y = mean(Y);
for y = 1:size(img, 1)
    for x = 1:size(img, 2)
        if central % compute central moments (mu)
            M_pq = M_pq + ((x-mean_X)^p * (y-mean_Y)^q * img(y, x));
        else % compute 'regular' moments (m)
            M_pq = M_pq + (x^p * y^q * img(y, x));
        end
    end
end

end

