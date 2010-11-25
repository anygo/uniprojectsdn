function img2 = my_interp2(x, y, img1)
    [yorig xorig] = size(img1);

    x(x<1) = 1; x(x>xorig) = xorig;
    y(y<1) = 1; y(y>yorig) = yorig;

 
    a = img1(floor(y), floor(x));
    b = img1(floor(y), ceil(x));
    c = img1(ceil(y), ceil(x));
    d = img1(ceil(y), floor(x));
    
    d_strich = x-x_floor;
    d_strichstrich = y-y_floor;
    
    X = img1(a).*(1-d_strich) + img1(d).*d_strich;
    Y = img1(b).*(1-d_strich) + img1(c).*d_strich;
    img2 = X.*(1-d_strichstrich) + Y.*d_strichstrich;
end

%% CRAP!!!