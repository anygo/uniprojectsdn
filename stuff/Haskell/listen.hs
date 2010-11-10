laenge([])= 0
laenge(h:tail) = 1 + (laenge(tail))

myconcat(x,[]) = x
myconcat([],x) = x
myconcat(h1:t1,y) = h1:(myconcat(t1,y))

myappend(liste,elem) = myconcat(liste,[elem])

listensumme([]) = 0
listensumme(h:t) = h + (listensumme(t))
