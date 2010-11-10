nation(deutsch).
nation(brite).
nation(schwede).
nation(daene).
nation(norweger).

farbe(rot).
farbe(weiss).
farbe(gelb).
farbe(blau).
farbe(gruen).

getraenk(tee).
getraenk(kaffee).
getraenk(milch).
getraenk(bier).
getraenk(wasser).

tier(hund).
tier(vogel).
tier(katze).
tier(pferd).
tier(fisch).

zigarette(pallmall).
zigarette(dunhill).
zigarette(marlboro).
zigarette(winfield).
zigarette(rothmanns).

nachbarschaft([]). % Basisfall der Rekursion
nachbarschaft([ [N,F,G,T,Z] | RestlicheHaeuser ]) :- 
	nachbarschaft(RestlicheHaeuser),	%Rekursion
	nation(N), \+ member([N,_,_,_,_] , RestlicheHaeuser),
	farbe(F), \+ member([_,F,_,_,_] , RestlicheHaeuser),
	getraenk(G), \+ member([_,_,G,_,_] , RestlicheHaeuser),
	tier(T), \+ member([_,_,_,T,_] , RestlicheHaeuser),
	zigarette(Z), \+ member([_,_,_,_,Z] , RestlicheHaeuser).

regel_a(X) :- member([brite,rot,_,_,_],X).
regel_b(X) :- member([schwede,_,_,hund,_],X).
regel_c(X) :- member([daene,_,tee,_,_],X).
regel_d(X) :- nextto([_,gruen,_,_,_],[_,weiss,_,_,_],X). 
regel_e(X) :- member([_,gruen,kaffee,_,_],X).
regel_f(X) :- member([_,_,_,vogel,pallmall],X).
regel_g([_,_,[_,_,milch,_,_],_,_]).
regel_h(X) :- member([_,gelb,_,_,dunhill],X).
regel_i([[norweger,_,_,_,_],_,_,_,_]).
regel_j(X) :- nextto([_,_,_,_,marlboro],[_,_,_,katze,_],X).
regel_j(X) :- nextto([_,_,_,katze,_],[_,_,_,_,marlboro],X).
regel_k(X) :- nextto([_,_,_,_,dunhill],[_,_,_,pferd,_],X).
regel_k(X) :- nextto([_,_,_,pferd,_],[_,_,_,_,dunhill],X).
regel_l(X) :- member([_,_,bier,_,winfield],X). 
regel_m(X) :- nextto([norweger,_,_,_,_],[_,blau,_,_,_],X).
regel_m(X) :- nextto([_,blau,_,_,_],[norweger,_,_,_,_],X).
regel_n(X) :- member([deutsch,_,_,_,rothmanns],X).
regel_o(X) :- nextto([_,_,wasser,_,_],[_,_,_,_,marlboro],X).
regel_o(X) :- nextto([_,_,_,_,marlboro],[_,_,wasser,_,_],X).



loesung(X) :-
	X = [_,_,_,_,_],
	regel_a(X),
	regel_b(X),
	regel_c(X),
	regel_d(X),
	regel_e(X),
	regel_f(X),
	regel_g(X),
	regel_h(X),
	regel_i(X),
	regel_j(X),
	regel_k(X),
	regel_l(X),
	regel_m(X),
	regel_n(X),
	regel_o(X),
	nachbarschaft(X).
