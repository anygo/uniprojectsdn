%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Interventional Medical Image Processing (IMIP)
% SS 2011
% (C) University Erlangen-Nuremberg
% Author: Attila Budai & Jan Paulus
% Exercise 5: Diffusion Registration 
% Complete the '???'-lines
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all;
clear all;

% image size
is = 128;

% step length
tau = 0.9;
% smoothness factor
alpha = 10;
% maximal number of iterations
maxIterations = 150;
% stopping threshold
threshold = 5.0;

% define the image grid omega
[X1, X2] = meshgrid(1:is, 1:is);

% initialize displacement field U with zero!
U1 = zeros(is,is);            
U2 = zeros(is,is);

% initialize force field F
F1 = zeros(is,is);            
F2 = zeros(is,is);

% temporal buffer
tmp = zeros(is, is);

% create the main diagonal vectors of the tridiagonal (I - 2*tau*A) matrix for the
% thomas algorithm
aav = ones(is*is,1);
bbv = ones(is*is-1,1);

% create tridiagonal matrix (I - 2*tau*A) represented by three vectors
aav = aav * (1+4*tau);  % main diagonal
bbv = bbv * (-2*tau);   % upper diagonal
ccv = bbv;              % lower diagonal

% images to register
R  = zeros(is);  % reference image
T  = zeros(is);  % template image

% square
R(X1 > 25 & X1 < 90 & X2 > 25 & X2 < 90)  = 200;
% circle
T((X1-64).^2 + (X2-64).^2 < 32^2) = 180; 

% normalize R -> [0,1]
normD = 1 / max(max(R));
R = R * normD;

% normalize T -> [0,1]
normD = 1 / max(max(T));
T = T * normD;

% show images before starting the registration
figure(1);
subplot(2,3,1);
colormap gray;
imagesc(R);
title('reference image R(X)')

subplot(2,3,2);
imagesc(T);
title('template image T(X)')

subplot(2,3,3);
imagesc(abs(T - R));
title('difference image T(X) - R(X)')
drawnow;

% start registration
for k = 1:maxIterations
  
  %   -------------------------------------------------------
  %  compute T on (X1-U1,X2-U2)
  %   -------------------------------------------------------
  fprintf('%d. iteration ...\n', k);
  
  Tk = interp2(X1, X2, T, X1-U1, X2-U2);
  
  % Values outside of T cannot be interpolated and are set to NaN by
  % interp2. They must be replaced with 0 for further computations.
  Tk(isnan(Tk)) = 0;
  
  oldU1 = U1;
  oldU2 = U2;
  
  Dk = Tk - R;

  % compute force in X1 direction
  F1 = Dk .* (Tk(:,[2:end,end]) - Tk(:,[1,1:end-1]));
  % smooth forces
  F1 = conv2(F1, [1 2 1; 2 4 2; 1 2 1]/16,'same');  
 
  dd1 = U1 + tau * F1 * alpha;
  
  dd1r = dd1';
  dd1rv = dd1r(:);
  
  dd1v = dd1(:);
  
  % solve for V1 and V2
  V1j1 = thomas(aav,bbv,ccv,dd1rv);
  V1j2 = thomas(aav,bbv,ccv,dd1v);
  
  tmp(:) = V1j1(:);
  tmp = tmp';
  V1j1(:) = tmp(:);
   
  % compute sum of inverse
  U1j12 = 1/2 * (V1j1 + V1j2);
  U1(:) = U1j12(:);

  % compute force in X2 direction
  F2 = Dk.*(Tk([2:end,end],:) - Tk([1,1:end-1],:));
  % smooth forces
  F2 = conv2(F2, [1 2 1; 2 4 2; 1 2 1]/16,'same');  

  dd2  = U2 + tau * F2 * alpha;
  
  dd2r = dd2';
  dd2rv = dd2r(:);
  
  dd2v = dd2(:);
  
  % solve for V1 and V2
  V2j1 = thomas(aav,bbv,ccv,dd2rv);
  V2j2 = thomas(aav,bbv,ccv,dd2v);
  
  tmp(:) = V2j1(:);
  tmp = tmp';
  V2j1(:) = tmp(:);
  
  % compute sum of inverse
  U2j12 = 1/2 * (V2j1 + V2j2); %%%%%%%%%%%%%%%%%%%%%%%%%%%%
  U2(:) = U2j12(:);
  
  change = 0.5 * (norm(U1-oldU1, 'fro') + norm(U2-oldU2, 'fro'));
  change
  
  % show currently deformed image
  subplot(2,3,4);
  imagesc(Tk); 
  s = sprintf('deformed template image T(X-U) (k=%d)', k);
  title(s)

  % show difference between currently deformed image and its reference
  subplot(2,3,5);
  imagesc(abs(Dk));
  s = sprintf('difference image T(X-U) - R(X) (change=%f)', change);
  title(s)
  drawnow;
  
  if(change <= threshold)
      break;
  end

end  % end k=1..maxIterations loop

% plot the deformation field symbolized by arrows
figure(2);
quiver(X1, X2, U1, U2);
title('deformation field U');
