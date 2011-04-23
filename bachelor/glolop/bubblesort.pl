% bubblesort.pl

bubblesort([],[]).
bubblesort([H|[H2|T]],L) :- H > H2, bubblesort([H|T],L2), append([H2,H],L2,L).
