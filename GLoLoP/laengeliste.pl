
lengtha([],0).
lengtha([Head|Tail],L) :- atom(Head), lengtha(Tail,L2), L is 1 + L2.
lengtha([Head|Tail],L) :- lengtha(Head,L1), lengtha(Tail,L2), L is L1 + L2.
