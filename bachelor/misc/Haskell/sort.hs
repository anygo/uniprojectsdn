-----
quickSort :: Ord a => [a] -> [a]
quickSort [] = []
quickSort [h] = [h]
quickSort (h:t) = (quickSort ([x | x <- t, x <= h])) ++ [h] ++ (quickSort ([x | x <- t, x > h]))
-----
selSort :: Ord a => [a] -> [a]
selSort [] = []
selSort l = (minimum l):(selSort (delfirst (minimum l) l))

delfirst _ [] = []
delfirst x (h:t) | x == h = t
				 | otherwise = (h:(delfirst x t))
-----
mergeSort :: Ord a => [a] -> [a]
mergeSort [] = []
-- TODO --
-----
insSort :: Ord a => [a] -> [a]
insSort [] = []
insSort l = foldl ins [] l

ins [] x = [x]
ins (h:t) x | x > h = h:(ins t x)
			| otherwise = (x:h:t)
-----

