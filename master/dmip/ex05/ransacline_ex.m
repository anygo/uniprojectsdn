%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Diagnostic Medical Image Processing (DMIP) 
% WS 2010/11
% Author: Attila Budai & Jan Paulus
% Exercise: RANSAC
% NOTE: Complete the '???' lines
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function ransacline_ex
    clear all;
    close all;
    clc;
    
    % Point amount for model estimation (minimum number according to
    % RANSAC), probability for outliers, probability for correct model
    mn = 2;
    oP = 0.2;
    cP = 0.9999;
    
    % Data points to fit
    pts = [0   0;
           1   1;
           2   2;
           3   3;
           3.2 1.9;
           4   4;
           10  1.8];
       
    % Draw estimated line considering all points   
    figure(1);
    hold on;
    title('RANSAC for line fitting');
    xlabel('x');
    ylabel('y');
    
    x = [-min(pts(:,1))-5 max(pts(:,1))+5];
    aMdl = fitline(pts);
    am = aMdl(1);
    at = aMdl(2);
    
    ay = x*am+at;
    
    plot(x,ay);
    plot(pts(:,1)',pts(:,2)','x');
       
    % Estimate best model using RANSAC
    [eMdl outl err] = commonransac(pts, mn, oP, cP, @fitline, @lineerror);
    
    % Draw estimated line and outliers
    em = eMdl(1);
    et = eMdl(2);
    
    ey = x*em+et;
    
    plot(x,ey,'r');
    plot(outl(:,1)',outl(:,2)','or');
    
    legend('Line using all points', 'Data ponts', 'RANSAC estimated line', 'Points used for model estimation', 'Location', 'NorthWest');
    
    err
    em
    et
    
    
end

function [bMdl mPts err] = commonransac(data, mn, oRelFr, cProb, mdlEstFct, errFct)
% RANSAC function to return best estimated model parameters
% 
%   [bMdl outl err] = commonransac(data, mn, oProb, cProb, mdlEstFct,
%   errFct) returns the best estimated model parameters bMdl, the used
%   sample points mPts for building the model and the error err of the
%   estimated model. Used parameters are data as data points, mn as minimum
%   number of data points required to build the model, oRelFr as relative
%   frequency of outliers, cProb as probability to randomly choose the best
%   estimated model in one of the iterations, mdlEstFct as function for
%   model estimation and err Fct as error function. data is assumed to
%   represent one data point in each row. mdlEstFct and errFct
%   require the following interfaces:
%   [mdl] = mdlEstFct(data)
%   [err] = errFct(mdl,data)
% 
    it = ceil(log(1-cProb)/log(1-(1-oRelFr)^mn));
    n = size(data,1);

    err = Inf;
    bMdl = [];
    mPts = [];
    
    for i=1:it
        indices = randperm(n);
        indices = indices(1:mn); % the first mn random indices for data
        mData = data(indices, :);
     
        cMdl = mdlEstFct(mData);

        lErr = errFct(cMdl, data);
        
        if(lErr < err)
            err = lErr;
            mPts = mData;
            bMdl = cMdl;
        end
    end
end

function [err] = lineerror(mdl,pts)
% Estimates the error for a line fitted through the data points pts using 
% model parameters mdl

    % line equation: y = mx + t
    m = mdl(1,1); % slope
    t = mdl(2,1); % offset
    thresh = 1;
    
    % reformulation: y - mx - t = 0
    % (distance from line)

    err = 0;
    for i = 1:size(pts,1) 
        x = pts(i,1);
        y = pts(i,2);
        distance = abs(y - m*x - t);
        
        if (distance > thresh)
            err = err+1;
            continue; % it is an outlier -> ignore it
        else
            err = err + distance^2;
        end
    end
    
end

function [mdl] = fitline(pts)
% Fits a line throug pts using least squares

    % Build components for eq:
    %
    %         |m|                      
    %  M    * |t| =  ptsY
    %
    % and solve it to get the line parameters m and t 

    % measurement matrix
    % |x1 1|         |y1|
    % |x2 1|         |y2|
    % |x3 1| * |m| = |y3|
    %  .  .    |t|    .
    %  .  .           .
    %  .  .           .
    
    
    M = ones(size(pts,1),2);
    
    M(:,1) = pts(:,1);

    ptsY = pts(:,2);
    
    % solve the crap
    mdl = pinv(M)*ptsY; % slope m and offset t 
end