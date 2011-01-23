% quicksort.pl

quicksort([],[]).
quicksort([Pivot|Unsorted],Sorted) :-
	partition(Pivot,Unsorted,Littles,Bigs),
	quicksort(Littles,LS),
	quicksort(Bigs,BS),
	append(LS,[Pivot|BS],Sorted).

partition(_,[],[],[]).
partition(Pivot,[Hunsorted|Tunsorted],Littles,[Hunsorted|Bigs]) :-
	Pivot =< Hunsorted, 
	!,
	partition(Pivot,Tunsorted,Littles,Bigs).
partition(Pivot,[Hunsorted|Tunsorted],[Hunsorted|Littles],Bigs) :-
	Pivot > Hunsorted,
	!,
	partition(Pivot,Tunsorted,Littles,Bigs).
