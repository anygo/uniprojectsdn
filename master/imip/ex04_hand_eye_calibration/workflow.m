%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Interventional Medical Image Processing (IMIP)
% SS 2011
% (C) University Erlangen-Nuremberg
% Author: Attila Budai & Jan Paulus
% Toolboxes: Camera Calibration Toolbox for Matlab, 
%            Hand-Eye calibration addon
% Exercise 4: Hand-Eye calibration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all;
clear all;
% memory

% set paths to the toolboxes
addpath /var/tmp/sidoneum/calib_toolbox_addon;
addpath /var/tmp/sidoneum/calib_toolbox_addon/example2;
addpath /var/tmp/sidoneum/toolbox_calib;

% switch to addon directory example2
cd /var/tmp/sidoneum/calib_toolbox_addon/example2;

% scale the grid: 2mm
dX = 2;
dY = 2;
est_dist = [1 1 1 1 0]';

% you want to see what the calibration is doing
doShow = 1;
% your images show severe distortion, such 
% as from an endoscope or extremely wide optics
isStronglyDistorted = 0;
% you want the system to give the best 
% results. This will try to detect wrong correspondences 
% and calibrate again
doIterateCalibration = 1;
% you want your views sorted in a way that the interstation movement is ideal for 
% hand-eye calibration, see [Tsai]
doSortHandEyeMovement = 0;

% runs the Camera Calibration Toolbox
calib_gui
% image dimension 600x800; 10 images
% Standard (all the images are stored in memory)
% Read images
% > Basename camera calibration images (without number nor suffix): 
% > Image format: ([]='r'='ras', 'b'='bmp', 't'='tif', 'p'='pgm', 'j'='jpg',
% 'm'='ppm') m

% pauses execution for 30 seconds or set break point
% pause(30)
autocalibrate
load robot_poses.mat
handeye
% hit any key to browse through the images

% Increasing System Swap Space
% http://www.mathworks.com/access/helpdesk/help/techdoc/index.html?/access/
% helpdesk/help/techdoc/matlab_prog/brh72ex-49.html&http://www.google.de/se
% arch?hl=de&q=matlab+add+more+memory+to+the+system&btnG=Suche&meta=