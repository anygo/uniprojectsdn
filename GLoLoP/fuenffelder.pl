% 5felder.pl

farbe(X) :- 
	X = 'a';
	X = 'b';
	X = 'c'.

felder(X) :-
	X is 'f1';
	X is 'f2';
	X is 'f3';
	X is 'f4';
	X is 'f5'.

nextto(f1,f2).
nextto(f1,f3).
nextto(f1,f5).
nextto(f2,f4).
nextto(f2,f3).
nextto(f4,f3).
nextto(f4,f5).
nextto(f5,f3).
nextto(X,Y) :-
	nextto(Y,X).

loesung(F1,F2,F3,F4,F5) :-
	farbe(F1),farbe(F2),farbe(F3),farbe(F4),farbe(F5),
	F1 \= F2, F1 \= F3, F1 \= F5,
	F2 \= F4, F2 \= F3,
	F4 \= F3, F4 \= F5,
	F5 \= F3.
