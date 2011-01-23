% Summe einer Liste

sum([],fehler).
sum([X|[]],SUM) :- SUM is X.
sum([Head|Tail],SUM) :- Tail \= [], sum(Tail,SUM2), SUM is Head+SUM2.
sum([Head|[]],Head).
