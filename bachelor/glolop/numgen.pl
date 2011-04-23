% numgen.pl

% Usage:
% numgen(FROM,TO,OUTPUT).

numgen(X,X,X) :- integer(X).
numgen(FROM,TO,FROM) :-
	FROM < TO.
numgen(FROM,TO,OUTPUT) :-
	FROM < TO, FROM2 is FROM+1,
	numgen(FROM2,TO,OUTPUT).
