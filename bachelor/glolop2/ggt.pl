% ggt.pl

ggT(0,N,N) :- !.
ggT(X,Y,T) :- ( (Y>X, M is Y, N is X) ; (M is X, N is Y) ), R is M-N, ggT(R,N,T).
