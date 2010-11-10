% fakultaet.pl

fak(0, 1).
fak(1, 1).
fak(X, Y) :- X<0, Y = false.
fak(X, Y) :- X>0, X2 is X-1, fak(X2,Y2), Y is X*Y2.
