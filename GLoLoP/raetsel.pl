% raetsel.pl

unternehmung(X) :-
	X = disco;
	X = orgel;
	X = kino;
	X = park.

loesung(A,B,C,D,E,F,G,H) :-
	unternehmung(A), unternehmung(B),
	unternehmung(C), unternehmung(D),	unternehmung(G), unternehmung(H),
	unternehmung(E), unternehmung(F),
	A = disco, B = G, C \= F, H = kino, F = orgel,
	A \= B, A \= C, A \= D,
	B \= C, B \=D,
	C \= D,
	E \= F, E \= G, E \= H,
	F \= G, F \= H,
	G \= H.

zahl(X) :- member(X, [0,1,2,3,4,5,6,7,8,9]).

rechnung(S,E,N,D,M,O,R,Y) :-
	zahl(S), zahl(E), zahl(N), zahl(D), zahl(M), zahl(O), zahl(R), zahl(Y),
	S \= E, S \= N, S \= D, S \= M, S \= O, S \= R, S \= Y,
	Send1000 is S*1000, Send100 is E*100, Send10 is N*10, Send1 is D, SEND is Send1000+Send100+Send10+Send1,
	More1000 is M*1000, More100 is O*100, More10 is R*10, More1 is E, MORE is More1000+More100+More10+More1,
	Money10000 is M*10000, Money1000 is O*1000, Money100 is N*100, Money10 is E*10, Money1 is Y, MONEY is Money10000+Money1000+Money100+Money10+Money1,
	MONEY is SEND+MORE.

abcd(A,B,C,D) :-
	zahl(A), zahl(B), zahl(C), zahl(D),
	ABCD1000 is A*1000, ABCD100 is B*100, ABCD10 is C*10, ABCD1 is D*1, ABCD is ABCD1000+ABCD100+ABCD10+ABCD1,
	ACBD1000 is A*1000, ACBD100 is C*100, ACBD10 is B*10, ACBD1 is D*1, ACBD is ACBD1000+ACBD100+ACBD10+ACBD1,
	DBAC1000 is D*1000, DBAC100 is B*100, DBAC10 is A*10, DBAC1 is C*1, DBAC is DBAC1000+DBAC100+DBAC10+DBAC1,
	DBAC is ABCD+ACBD.
