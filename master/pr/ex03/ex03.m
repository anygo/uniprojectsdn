%% cleanup
close all;
clear all;

%% input data
S1 = [1 1; 1 2; 2 1; 3 1];
S2 = [2 4; 2 3; 3 2; 4 2];
to_classify = [2 6; 6 2];
Sall = [S1; S2]';

%% transformed (lhs*a = rhs) 
lhs = [1 1 1; 1 2 4; 4 2 1; 9 3 1; 4 8 16; 4 6 9; 9 6 4; 16 8 4];
rhs = [0; 0; 0; 0; 1; 1; 1; 1];

%% pseudoinverse of lhs (a = inv(lhs)*rhs)
[U S V] = svd(lhs);
Sinv = zeros(size(S));
Sinv(1,1) = 1/S(1,1);
Sinv(2,2) = 1/S(2,2);
Sinv(3,3) = 1/S(3,3);
Sinv = Sinv';
lhs_inv = V*Sinv*U';

%% least square solution
a = lhs_inv*rhs;

%% contour zeugs
[X,Y] = meshgrid(0:0.1:6, 0:0.1:6);
Z = a(1,1).*X.*X + a(2,1).*X.*Y + a(3,1).*Y.*Y;

%% plot 
axis([0 6 0 6])
contour(X,Y,Z, 0.5, 'black')
hold on;
scatter(S2(:,1), S2(:,2), 'or');
scatter(S1(:,1), S1(:,2), '+b');
scatter(to_classify(:,1), to_classify(:,2), 'dm');
hold off;