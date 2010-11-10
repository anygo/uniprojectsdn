main = interact stringReverse

stringReverse [] = []
stringReverse (h:t) = (stringReverse t) ++ [h]
