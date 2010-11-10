istPrim(x) = if (x < 0) then 0 else istPrim2(x,(x-1))
istPrim2(x,y) = if (y < 2) then 1 else if (mod x y == 0) then 0 else istPrim2(x,(y-1))

myconcat(x,[]) = x
myconcat([],x) = x
myconcat(h1:t1,y) = h1:(myconcat(t1,y))

myappend(liste,elem) = myconcat(liste,[elem])

primFilter([]) = []
primFilter(h:t) = primFilter2([],h:t)
primFilter2(newlist,[]) = newlist
primFilter2(newlist,h:t) = if (istPrim(h) == 0) then myconcat(myappend(newlist,h),primFilter(t)) else myconcat(newlist,primFilter(t))

primListe(start,ende) = primListe2([],start,ende)
primListe2(newlist,start,ende) = if (start > ende) then newlist else primListe3(newlist,start,ende)
primListe3(newlist,start,ende) = if (istPrim(start) == 1) then myconcat(myappend(newlist,start),primListe2([],(start+1),ende)) else myconcat(newlist,primListe2([],(start+1),ende))
