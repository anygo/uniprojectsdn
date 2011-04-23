% Blatt 8


% --- rek_length/2
rek_length([], N) :- N is 0.
rek_length([H | T], N) :- atom(H), rek_length(T, N2), N is 1 + N2.
rek_length([H | T], N) :- rek_length(H, N1), rek_length(T, N2), N is N1 + N2.
% done


% --- my_flatten/2
my_flatten([], L) :- L = [].
my_flatten(X, L) :- atom(X), L = [X].
my_flatten([H | T], L) :- my_flatten(H, L1), my_flatten(T, L2), append(L1, L2, L).
% done


% --- remove_first/3
remove_first([],L,L) :- !.
remove_first([H|T1],[H|T2], L) :-
	remove_first2(T1,T2,L).
remove_first([H1|T1],[H2|T2],[H2|L]) :-
	remove_first([H1|T1],T2,L).

remove_first2([],L,L) :- !.
remove_first2([H|T1],[H|T2],L) :-
	remove_first2(T1,T2,L).
% done
