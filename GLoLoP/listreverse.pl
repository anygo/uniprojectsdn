% listreverse.pl

reverse([],[]).
reverse([H|T],L) :- reverse(T,L1), append(L1,[H],L).
