% blabla

byCar(auckland,hamilton).
byCar(hamilton,raglan).
byCar(valmont,saarbruecken).
byCar(valmont,metz).

byTrain(metz,frankfurt).
byTrain(saarbruecken,frankfurt).
byTrain(metz,paris).
byTrain(saarbruecken,paris).

byPlane(frankfurt,bangkok).
byPlane(frankfurt,singapore).
byPlane(paris,losAngeles).
byPlane(bangkok,auckland).
byPlane(losAngeles,auckland).


travel(A,B) :- byCar(A,B); byTrain(A,B); byPlane(A,B).
travel(A,B) :- (byCar(A,C); byTrain(A,C); byPlane(A,C)), travel(C,B).

