% quicksort_olo.pl

quicksort(Xs,Ys) :- quicksort_dl(Xs,Ys-[]).

quicksort_dl([],Xs-Ys).

quicksort_dl([X|Xs],Ys-Zs) :-
	partition(X,Xs,Littles,Bigs),
	quicksort_dl(Littles,Ys-[X|Yls]),
	quicksort_dl(Bigs,Yls-Zs).

partition(_, [], [], []).
partition(X, [Y | Xs], [Y | Ls], Bs) :-
   X > Y,
      !,
	     partition(X, Xs, Ls, Bs).
partition(X, [Y | Xs], Ls, [Y | Bs]) :-
		    X =< Y,
			   !,
			      partition(X, Xs, Ls, Bs).

