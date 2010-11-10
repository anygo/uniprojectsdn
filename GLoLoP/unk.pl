% unk.pl

unk([], []).
unk([H|T], L) :- unk(H, A), unk(T, B), append(A, B, L).
unk(X, [X]).
