%regeln

s(salat).
s(kein_salat).
b(brot).
b(kein_brot).
e(eis).
e(kein_eis).

regel1(kein_brot, _, eis). 
regel1(brot, _, _).

regel2(brot, kein_salat, eis).
regel2(kein_brot, _, _).
regel2(_, _, kein_eis).

regel3(kein_brot, salat, kein_eis).
regel3(brot, _, _).
regel3(_, kein_salat, _).

perfektes_essen(S, B, E) :- s(S), b(B), e(E), regel1(B, S, E), regel2(B, S, E), regel3(B, S, E).
