% essen.pl

zutat(salat).
zutat(ksalat).
zutat(brot).
zutat(kbrot).
zutat(eis).
zutat(keis).

regel1(_,kbrot,eis).
regel1(_,brot,_).
regel2(ksalat,brot,eis).
regel2(_,_,keis).
regel2(_,kbrot,_
regel3(salat,kbrot,keis).
regel3(_,brot,_).
regel3(ksalat,_,_).

perfekt(S,B,E) :- regel1(S,B,E), regel2(S,B,E), regel3(S,B,E).

start :- findall((B,E,S),perfekt(B,E,S),RESULT),write('Die blabla sind:'),write(RESULT).
