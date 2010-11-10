% kreuzkringel.pl

gueltig([]).
gueltig(o).
gueltig(X) :- append(Y, [x,E,x|R],X), gueltig(Y), gueltig(R), gueltig(E).

