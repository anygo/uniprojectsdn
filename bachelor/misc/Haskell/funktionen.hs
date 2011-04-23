quadrat(x) = x*x

betrag(x) = if x >= 0 then x else -x

wurzelbla(a,b,c) | (b*b - 4*a*c >= 0) = sqrt(b*b - 4*a*c)
				 | otherwise = -1 --FEHLERdarstellung?!

quadGleichung(a,b,c) | (wurzelbla(a,b,c) == 0) = [((-b)/(2*a))]
					 | (wurzelbla(a,b,c) < 0) = []
					 | otherwise = (-b + sqrt(b*b - 4*a*c)/2*a):(-b - sqrt(b*b - 4*a*c)/2*a):[]

fak(0) = 1
fak(1) = 1
fak(x) = if (x < 0) then -1 else x * (fak(x-1))
