% efface.pl

efface(X,[X|T],T).
efface(X,[H|T],[H|Teff]) :- efface(X,T,Teff).

del_all(_,[],[]).
del_all(X,[X|T],L) :- del_all(X,T,L), !.
del_all(X,[H|T],[H|L]) :- del_all(X,T,L), !.

del_last(_,[],[]).
del_last(X,[X|T],T) :- \+ member(X,T), !.
del_last(X,[H|T],[H|L]) :- del_last(X,T,L).

