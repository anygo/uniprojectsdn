% ableitung.pl

regel1(a,_,_).
regel2(a,a,_).
regel3(_,a,a).
regel4(a,b,_).
regel5(_,a,b).
regel6(b,c,_).
regel7(_,b,c).

test(A,B,C) :- regel1(A,B,C); regel2(A,B,C); regel3(A,B,C); regel4(A,B,C); regel5(A,B,C); regel6(A,B,C); regel7(A,B,C).
