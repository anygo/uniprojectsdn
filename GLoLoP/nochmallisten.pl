% nochmallisten.pl

my_member(_,[],false).
my_member(X,[X|_],true) :- !.
my_member(X,[_|T],Y) :- my_member(X,T,Y).

flatten([],[]).
flatten([H|T],L) :- flatten(H,L1), flatten(T,L2), append(L1,L2,L).
flatten(X,[X]).

list_length([],0).
list_length([_|T],L) :- list_length(T,L2), L is 1+L2.

rek_length([],0).
rek_length(X, N) :- flatten(X, L), list_length(L, N).

remove_first(_, [], []).
remove_first([], L, L). 
remove_first([H|T], [H|T1], T2) :- remove_first(T, T1, T2).
remove_first([H|T], [H1|T1], [H1|T2]) :- remove_first([H|T], T1, [H1|T2]).
