% fibnacci.pl

% berechnet die N'te Fibonacci-Zahl


fib(0,1).
fib(1,1).
fib(X,Y) :- X>1, X2 is X-1, X3 is X-2, fib(X2,Y2), fib(X3,Y3), Y is Y2+Y3.
