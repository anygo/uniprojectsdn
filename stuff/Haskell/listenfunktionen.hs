f x = x^3

mymap f [] = []
mymap f (h:t) = (f h):(mymap f t)

myfoldr f zahl [] = zahl
myfoldr f zahl (h:t) = f h (myfoldr f zahl t)

myfoldl f zahl [] = zahl
myfoldl f zahl (h:t) = myfoldl f (f zahl h) t

myreverse [] = []
myreverse (h:t) = myreverse t ++ [h]

myzip [] _ = []
myzip _ [] = []
myzip (h1:t1) (h2:t2) = [(h1,h2)] ++ (myzip t1 t2)

myzipWith f [] _ = []
myzipWith f _ [] = []
myzipWith f (h1:t1) (h2:t2) = [f h1 h2] ++ myzipWith f t1 t2
