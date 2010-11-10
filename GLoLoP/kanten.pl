% kanten

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

kante(a,b,2).
kante(a,c,5).
kante(a,d,7).
kante(c,e,1).
kante(b,f,9).
kante(e,f,2).
kante(d,f,5).
kante(f,g,7).
kante(h,i,7).
kante(h,j,6).
kante(j,k,1).
kante(j,l,2).
kante(j,m,5).
kante(i,j,1).
kante(g,h,18).

erreichbar(X,Y) :- kante(X,Y,_).
erreichbar(X,Y) :- kante(X,Z,_), erreichbar(Z,Y).

erreichbar(X,Y,Pfad,Kosten) :- kante(X,Y,K), Pfad = [X,Y], Kosten is K.
erreichbar(X,Y,Pfad,Kosten) :- kante(X,Z,K1), erreichbar(Z,Y,Pfad2,K2), Pfad = [X|Pfad2], Kosten is K1+K2. 

getlist(X,Y,L) :- findall([Kosten|Pfad], erreichbar(X,Y,Pfad,Kosten), L).

getminstart(X,Y) :- getlist(X,Y,L), getmin(L, 10000, [], _, _).
getmaxstart(X,Y) :- getlist(X,Y,L), getmax(L, 0, [], _, _).

getmin([],Kosten,Pfad,Kosten,Pfad) :- write('Minimale Kosten: '), write(Kosten), nl, write('Pfad: '), write(Pfad), fail.
getmin([[K|P]|Rest],Kosten,_,MinKosten,MinPfad) :- minimum(K,Kosten,MIN), MIN = K, getmin(Rest,MIN,P,MinKosten,MinPfad). 
getmin([[K|_]|Rest],Kosten,P,MinKosten,MinPfad) :- minimum(K,Kosten,MIN), MIN \= K, getmin(Rest,MIN,P,MinKosten,MinPfad).

getmax([],Kosten,Pfad,Kosten,Pfad) :- write('Maximale Kosten: '), write(Kosten), nl, write('Pfad: '), write(Pfad), fail.
getmax([[K|P]|Rest],Kosten,_,MaxKosten,MaxPfad) :- maximum(K,Kosten,MAX), MAX = K, getmax(Rest,MAX,P,MaxKosten,MaxPfad).
getmax([[K|_]|Rest],Kosten,P,MaxKosten,MaxPfad) :- maximum(K,Kosten,MAX), MAX \= K, getmax(Rest,MAX,P,MaxKosten,MaxPfad). 

minimum(A,B,L) :- 	A =< B, L is A; 
					B < A, L is B.

maximum(A,B,L) :-	A >= B, L is A;
					B > A, L is B.


minimumliste([],In,In).
minimumliste([H|T],In,Out) :- minimum(H,In,X), minimumliste(T,X,Out).
