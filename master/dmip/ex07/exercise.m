% dmip exercise sheet 7
close all;
clear all;
clc;

%% exercise 1 - Axis-Angle Representation
% definition of rotation axis t
t = rand(3,1)*2-1;
t = t/norm(t);

% generate random points
points = rand(3,5);

% plot coordinate- and rotation-axes
hold on; 
plot3([0 1], [0 0], [0 0], 'black'); 
plot3([0 0], [0 1], [0 0], 'black'); 
plot3([0 0], [0 0], [0 1], 'black'); 
plot3([0 t(1)], [0 t(2)], [0 t(3)], 'r'); 
hold off;

for theta_rad = 0.1:0.2:2*pi
    theta_deg = theta_rad * 180 / pi;
    disp(['angle in degrees: ' num2str(theta_deg)]);

    % set up rotation matrix R with Rodrigues' formula
    skew_matrix = [0 -t(3) t(2); t(3) 0 -t(1); -t(2) t(1) 0];
    R = t*t' + (eye(3) - t*t')*cos(theta_rad) + skew_matrix*sin(theta_rad);

    % check if R is a valid rotation matrix
    if abs(det(R)-1) > 1e-4
        disp('determinant is not equal to 1');
    end
    
    % transform it back to axis-angle representation
    e = eig(R);
    theta_rad_transformed = acos((sum(e)-1)/2);
    error_theta = abs(pi-theta_rad_transformed) - abs(pi-theta_rad);
    t_transformed = null(R-eye(3));
    tmp = abs(t) ./ abs(t_transformed);
    tmp = tmp/mean(tmp);
    mse = norm(tmp - 1);
    disp(['mse of rotation axes (ignoring scaling): ' num2str(mse)]);
    disp(['Difference of rotation angle: ' num2str(error_theta)]);
    
    % rotate
    rotated = zeros(size(points));
    for i = 1:size(points,2)
        rotated(:,i) = R*points(:,i);
    end
    hold on;
    scatter3(rotated(1,:), rotated(2,:), rotated(3,:), 'r');
    hold off;
    axis equal;
    drawnow;
end

%% exercise 2 - Quaternions
% TODO

%% exercise 3 - Euler Angle Representation
% angles
tx = 1;
ty = 0.7;
tz = 2.2;

Rx = [1 0 0; 0 cos(tx) -sin(tx); 0 sin(tx) cos(tx)];
Ry = [cos(ty) 0 sin(ty); 0 1 0; -sin(ty) 0 cos(ty)];
Rz = [cos(tz) -sin(tz) 0; sin(tz) cos(tz) 0; 0 0 1];

R1 = Rz*Ry*Rx;
R2 = Rx*Ry*Rz;

ty1 = asin(R1(1,3));
tz1 = acos(R1(1,1)/cos(ty1));
tx1 = asin(R1(2,3)/cos(ty1));

ty2 = asin(R2(1,3));
tz2 = acos(R2(1,1)/cos(ty1));
tx2 = asin(R2(2,3)/cos(ty1));
