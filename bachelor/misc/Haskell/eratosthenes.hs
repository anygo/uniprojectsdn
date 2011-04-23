istPrim(x) = if (x < 0) then 0 else istPrim2(x,(x-1))
istPrim2(x,y) = if (y < 2) then 1 else if (mod x y == 0) then 0 else istPrim2(x,(y-1))

eratosthenes = [ x | x <- [0..], istPrim x == 1]
