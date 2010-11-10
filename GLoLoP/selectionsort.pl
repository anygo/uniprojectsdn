# vim:ft=prolog

% selectionsort.pl

selectionsort([],[]).
selectionsort([H|T],Sorted) :-
	findmax(T,H,MAX),
	delfirst(MAX,[H|T],Rest),
	selectionsort(Rest,Sorted2),
	append(Sorted2,[MAX],Sorted).

findmax([],CUR,CUR) :- !.
findmax([H|T],CUR,MAX) :-
	H >= CUR,
	findmax(T,H,MAX).
findmax([H|T],CUR,MAX) :-
	H < CUR,
	findmax(T,CUR,MAX).

delfirst(MAX,[MAX|T],T) :- !.
delfirst(MAX,[H|T],L) :-
	delfirst(MAX,T,L2),
	append([H],L2,L).
