% sudoku_meins.pl - wird ja eh wieder nicht funktionieren...

valid(X) :- X is 1;
			X is 2;
			X is 3;
			X is 4;
			X is 5;
			X is 6;
			X is 7;
			X is 8;
			X is 9.

genlist(A) :- genlist([],A,0).
genlist(Output,Output,9) :- !.
genlist(A,Output,Laenge) :- valid(X), \+ member(X,A), append(A,[X],C), Laenge2 is Laenge+1, genlist(C,Output,Laenge2).

del_of_list(X,[X|L],L) :- !.
del_of_list(X,[Irgendwas|Rest],Output) :- del_of_list(X,Rest,Y), append([Irgendwas],Y,Output).

gennums([1,2,3,4,5,6,7,8,9]).
