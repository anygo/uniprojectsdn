% ordered.pl

ordered([]).
ordered([X]) :- integer(X).
ordered([H|[H2|T]]) :- H < H2, ordered([H2|T]).
