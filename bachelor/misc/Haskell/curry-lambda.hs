--mycurry :: ((a,b) -> c) -> (a -> b -> c)
--mycurry f (a,b) = f (a b)

--myuncurry :: (a -> b -> c) -> ((a,b) -> c)
-- versteh ich nicht... ;-) 



-- Testfunktion --
f x = 2*x*x*x + x*x + x + 1
------------------

spiegelX :: Num a => (a -> a) -> (a -> a)
spiegelX f e = negate (f e)

spiegelY :: Num a => (a -> a) -> (a -> a)
spiegelY f e = f (negate e)
