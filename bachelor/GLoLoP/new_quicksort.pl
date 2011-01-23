% new_quicksort.pl

quicksort([],[]).
quicksort([Pivot|Tail],Sorted) :-
	partition(Pivot,Tail,Littles,Bigs),
	quicksort(Littles,LS),
	quicksort(Bigs,BS),
	append(LS,[Pivot|BS],Sorted).

partition(_,[],[],[]).
partition(Pivot,[H|Tail],[H|Littles],Bigs) :-
	Pivot >= H,
	partition(Pivot,Tail,Littles,Bigs).
partition(Pivot,[H|Tail],Littles,[H|Bigs]) :-
	Pivot < H,
	partition(Pivot,Tail,Littles,Bigs).
