% listen.pl

flatten([], X) :- X = ''.
flatten(E, X) :- X = E.
flatten([E|L], X) :- J = flatten(E), K = flatten(L), X is J + K.
