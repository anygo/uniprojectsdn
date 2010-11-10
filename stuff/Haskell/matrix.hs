skalarprod v1 v2 = sqrt (foldl (+) 0 (map (^2) (zipWith (+) v1 v2)))

betrag v = sqrt (foldl (+) 0 (map (^2) v))

matrixadd (lastLine1:[]) (lastLine2:[]) = [(zipWith (+) lastLine1 lastLine2)]
matrixadd (m1Line:m1Tail) (m2Line:m2Tail) = [(zipWith (+) m1Line m2Line)] ++ (matrixadd m1Tail m2Tail)


-- matrixmul (gefrickel) --------------------------START

-- eine Spalte aus Matrix holen
getC (h:[]) column = [(h !! column)]
getC (h:t) column = [(h !! column)] ++ getC t column

-- eine Zeile aus Matrix holen
getL m line = m !! line

-- eine Zeile der Ergebnismatrix erzeugen
setL m1 m2 line lineLength tmpColumn | tmpColumn <= lineLength = [(foldl (+) 0 (zipWith (*) (getL m1 line) (getC m2 tmpColumn)))] ++ (setL m1 m2 line lineLength (tmpColumn+1))
									 | otherwise = []

-- Spaltenanzahl der Ergebnismatrix bestimmen
getLineLength m1 m2 = length (getL m2 0)

-- Zeilenanzahl der Ergebnismatrix bestimmen
getColLength m1 m2 = length (getC m1 0)

matrixmul m1 m2 = matrixmulHELP m1 m2 0
matrixmulHELP m1 m2 line | line <= (getColLength m1 m2) = [(setL m1 m2 line (getColLength m1 m2) 0)] ++ [(setL m1 m2 (line+1) (getColLength m1 m2) 0)]
						 | otherwise = []

-- irgendwo is da der Wurm drin :/
-- scheiss gefrickel :)
-- matrixmul (gefrickel) ----------------------------END

