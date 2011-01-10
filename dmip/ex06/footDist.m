function [index] = footDist(point,Po,Pt,angle)

Xc = point(1);
Yc = point(2);

angle = (angle*pi)/180;
rotMat = [cos(angle), -sin(angle); sin(angle), cos(angle)];
Po = (rotMat*Po);
Pt = (rotMat*Pt);

Xa = Po(1);
Ya = Po(2);
  
Xb = Pt(1);
Yb = Pt(2);
  
numerator = (-(Xc-Xa)*(Yb-Ya)) + ((Yc-Ya)*(Xb-Xa));
denominator = ((Xb-Xa)^2)+((Yb-Ya)^2);

  
X = (-(Yb-Ya));
Y = (Xb-Xa);
  
scale = numerator/denominator; 

direction = [scale*X,scale*Y];
foot = point - direction;
index = norm(foot-Po');
