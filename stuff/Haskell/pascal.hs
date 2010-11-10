bin 0 0 = 1
bin n k | n == k = 1
bin n 0 = 1 
bin n k = bin (n-1) (k-1) + bin (n-1) k

getPascalLine 0 _ = []
getPascalLine 1 _ = [1]
getPascalLine x 0 = [1]
getPascalLine x y = [(bin x y)] ++ getPascalLine x (y-1)

pascal = pascal2 1 1
pascal2 a b = [(getPascalLine a b)] ++ pascal2 (a+1) (b+1)
