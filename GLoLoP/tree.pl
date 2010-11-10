% tree.pl (Aufgabe 9.3)

tree_empty(nil).

tree_create(V,tree(nil,nil,V)).

tree_getv(tree(_,_,V),V).
tree_getlb(tree(L,_,_),L).
tree_getrb(tree(_,R,_),R).

tree_setv(tree(L,R,_),VN,tree(L,R,VN)).
tree_setlb(tree(_,R,V),LN,tree(LN,R,V)).
tree_setrb(tree(L,_,V),RN,tree(L,RN,V)).

tree_is(tree(_,_,_)).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

max(A,B,X) :- 
	A >= B, X is A;
	A < B, X is B.

treeheight(nil,0) :- !.
treeheight(tree(T1,T2,_),L) :- treeheight(T1,L1), treeheight(T2,L2), max(L1,L2,ADD), L is 1+ADD.

treetrav(T,pre) :- trav_pre(T).
treetrav(T,in) :- trav_in(T).
treetrav(T,post) :- trav_post(T).

trav_pre(X) :- tree_empty(X).
trav_pre(T) :-
	tree_getlb(T,L),
	tree_getrb(T,R),
	tree_getv(T,V),
	write(V),
	write(' '),
	trav_pre(L),
	trav_pre(R).

trav_in(X) :- tree_empty(X).
trav_in(T) :-
	tree_getlb(T,L),
	tree_getrb(T,R),
	tree_getv(T,V),
	trav_in(L),
	write(V),
	write(' '),
	trav_in(R).

trav_post(X) :- tree_empty(X).
trav_post(T) :-
	tree_getlb(T,L),
	tree_getrb(T,R),
	tree_getv(T,V),
	trav_post(L),
	trav_post(R),
	write(V),
	write(' ').

tree_insert(T,V,NT) :-
	tree_empty(T),
	tree_create(V,NT).
tree_insert(T,V,NT) :-
	tree_getv(T,VT),
	V < VT,
	tree_getlb(T,L),
	tree_insert(L,V,NL),
	tree_setlb(T,NL,NT).
tree_insert(T,V,NT) :-
	tree_getv(T,VT),
	V >= VT,
	tree_getrb(T,R),
	tree_insert(R,V,NR),
	tree_setrb(T,NR,NT).

tree_find(T,V,T) :- tree_getv(T,V), write('YAAAAAYYYYY!!! '), write(V), nl, write('Hier ist der Wert.. YAHYYYYAA!! '), write(T).
tree_find(T,V,R) :-
	tree_getv(T,VT),
	V < VT,
	tree_getlb(T,L),
	tree_find(L,V,R).
tree_find(T,V,R) :-
	tree_getv(T,VT),
	V >= VT,
	tree_getrb(T,R),
	tree_find(R,V,R).
