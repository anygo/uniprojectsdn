% aufgabe8-3.pl

rek_length([],0).
rek_length(X,1) :- atom(X), X \= [].
rek_length([H|T],L) :- rek_length(H,L1), rek_length(T,L2), L is L1+L2.

% Inkompetente Vollidioten - Musterloesung ist Schrott!!!
remove_first(_,[],[]).
remove_first([],X,X).
remove_first([H|T],[H|T2],L) :- remove_first(T,T2,L).
remove_first([H|T],[H2|T2],[H2|L]) :- remove_first([H|T],T2,L).

my_flatten([],[]) :- !.
my_flatten(X,[X]) :- atom(X), !.
my_flatten([H|T],L) :- my_flatten(H,L1), my_flatten(T,L2), append(L1,L2,L).
