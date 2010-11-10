% aufgabe6-2.pl

knoten(a). 
knoten(b).
knoten(c). 
knoten(d).
knoten(e). 
knoten(f).
knoten(g). 
knoten(h).
knoten(i). 
knoten(j).
knoten(k). 
knoten(l).
knoten(m). 

kante(a,b).
kante(a,c). 
kante(a,d).
kante(c,e). 
kante(b,f).
kante(e,f). 
kante(d,f).
kante(f,g). 
kante(g,h).
kante(h,i). 
kante(h,j).
kante(i,j). 
kante(j,k).
kante(j,l). 
kante(j,m).

kosten(a,b,2).
kosten(a,c,5). 
kosten(a,d,7).
kosten(c,e,1). 
kosten(b,f,9).
kosten(e,f,2). 
kosten(d,f,5).
kosten(f,g,7). 
kosten(g,h,18).
kosten(h,i,7). 
kosten(h,j,6).
kosten(i,j,1). 
kosten(j,k,1).
kosten(j,l,2). 
kosten(j,m,5).

erreichbar(A,A,0,[A]) :- !.
erreichbar(A,B,K,[A,B]) :-
	knoten(A),
	knoten(B),
	kosten(A,B,K).
erreichbar(A,B,K,P) :-
	knoten(A),
	knoten(B),
	knoten(Y),
	kosten(A,Y,K1),
	erreichbar(Y,B,K2,P2),
	P = [A|P2],
	K is K1+K2.

minimum(X,Y,X) :- X =< Y, !.
minimum(X,Y,Y) :- X > Y, !.

maximum(X,Y,X) :- X >= Y, !.
maximum(X,Y,Y) :- X < Y, !.

% L im Format: [[Kosten1, [Pfad1]], [Kosten2, [Pfad2]], ...]
wege2liste(A,B,L) :-
	findall([K,P],erreichbar(A,B,K,P),L).

loesung(A,B,K,P,X) :-
	(X = min,
	wege2liste(A,B,L),
	findmin(L,1000,[failure],K,P));
	(X = max,
	wege2liste(A,B,L),
	findmax(L,-1,[failure],K,P)).


findmax([],K,P,K,P) :- !.
findmax([[Knew,Pnew]|T],K,P,Kout,Pout) :-
	(maximum(Knew,K,Knew),
	findmax(T,Knew,Pnew,Kout,Pout));
	(maximum(Knew,K,K),
	findmax(T,K,P,Kout,Pout)).

findmin([],K,P,K,P) :- !.
findmin([[Knew,Pnew]|T],K,P,Kout,Pout) :-
	(minimum(Knew,K,Knew),
	findmin(T,Knew,Pnew,Kout,Pout));
	(minimum(Knew,K,K),
	findmin(T,K,P,Kout,Pout)).
