% damen.pl

acht_damen(Stellung) :-
		length(Stellung, 8),
		numlist(1, 8, Positionen),
		konstruiere_stellung(Positionen, Positionen, Stellung),
		gueltige_stellung(Stellung),
		write(Stellung).

konstruiere_stellung(_,_,[]).
konstruiere_stellung([X1|RestXListe], YListe, [[X1, Y]|Rest]) :-
		select(Y, YListe, RestYListe),
		konstruiere_stellung(RestXListe, RestYListe, Rest).

gueltige_stellung([]).
gueltige_stellung([[X1, Y1]|Rest]) :-
		schlaegt_nicht(X1,Y1,Rest),
		gueltige_stellung(Rest).

schlaegt_nicht(_,_,[]).
schlaegt_nicht(X1,Y1,[[X2,Y2]|Rest]) :-
		X1 =\= X2,
		Y1 =\= Y2,
		DeltaX is abs(X1-X2),
		DeltaY is abs(Y1-Y2),
		DeltaX =\= DeltaY,
		schlaegt_nicht(X1,Y1,Rest).
