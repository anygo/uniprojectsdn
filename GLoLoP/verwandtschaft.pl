% verwandtschaft.pl - absoluter Kaeser ;)

mann(X) :-
	X = peter;
	X = fritz;
	X = franz;
	X = axel;
	X = moritz.

frau(X) :-
	X = christine;
	X = bertraud;
	X = anna.

vater(peter,fritz).
vater(peter,christine).
vater(peter,moritz).

mutter(anna,betraud).

vater(franz,peter).
vater(axel,franz).

elternteil(A,B) :-
	vater(A,B);
	mutter(A,B).

verwandt(A,B,geschwister) :- 
	vater(Y,A), vater(Y,B);
	mutter(Y,A), mutter(Y,B).
verwandt(A,B,kind) :-
	elternteil(A,B).
verwandt(A,B,elternteil) :-
	elternteil(B,A).
verwandtgrad(A,B,N) :-
	elternteil(A,B), N is 1;
	elternteil(B,A), N is 1;
	verwandt(A,B,geschwister), N is 0;
	elternteil(A,C), elternteil(C,B), N is 2;
	elternteil(B,C), elternteil(C,A), N is 2.
