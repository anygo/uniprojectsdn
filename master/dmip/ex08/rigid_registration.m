%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Diagnostic Medical Image Processing (DMIP) 
% WS 2010/11
% Author: Attila Budai & Jan Paulus
% Exercise: Rigid registration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function rigid_registration()
    close all;
    clear all;
    clc;

    % Set initial deformation
    rotation = 15;
    shiftx = 10;
    shifty = 5;

    % image size
    is = 256;
    pad = 32;

    % generate phantom image
    I1 = phantom(is);
    I1 = padarray(I1,[pad pad]);
    g = fspecial('gaussian', 15, 8 );
    I1 = imfilter(I1, g, 'same');
    I2 = transform(I1, rotation, shiftx, shifty);

    %Initial visualization
    figure(1);
    colormap gray;
    subplot(2,3,1);
    imagesc(I1);
    title('original image');
    subplot(2,3,2);
    imagesc(I2);
    title('transformed image');
    startingpos = [ 10 10 10 ];
    I3 = transform(I2, startingpos(1), startingpos(2), startingpos(3));
    subplot(2,3,3);
    imagesc(I3);
    title('starting image');


    %Otimization using fminsearch()
    values = [];
    [x, fval] = fminsearch(@(x) similarity(x, I1, I2), startingpos, optimset('MaxIter', 100));
    
    %Similarity measurement function called by fminsearch
    function d = similarity(pos, static, moving)
        pos
        %Apply current transformation
        moved = transform(moving, pos(1), pos(2), pos(3) );
        is = size(static);

        %Calculate distance function: absolute distance        
        diff = abs(static-moved);
        %d = sum(sum(diff))/(is(1)*is(2));

        %Calculate distance function: SSD
        %sd = sum(sum((static-moved).^2))/(is(1)*is(2));

        %Calculate distance function: Mutual Information
        jh = jointHistogram(im2uint8(static), im2uint8(moved));
        jointEntropy = -sum(sum(jh.*log2(jh+(jh==0))));
        d = -(marginalEntropy(jh') + marginalEntropy(jh) - jointEntropy) ; % mutual information

        values = [values d];

        %Visualization after each iteration
        subplot(2,3,4);
        imagesc(moved);
        title('moving image');    
        subplot(2,3,5);
        imagesc(diff);
        title('difference image');
        subplot(2,3,6);
        plot(values);
        title('changes in distance');
        drawnow;
    end

    %Function to apply transformation
    function out = transform(in, rot, shiftx, shifty)

        %Rotate image with "rot" degree and keeping the original image size
        out = imrotate(in, rot, 'crop');
        out(isnan(out)) = 0;
           
        %Shift in x and y direction with "shiftx" and "shifty" pixels       
        [X Y] = meshgrid(1:size(out,1), 1:size(out,2));
        out = interp2(X, Y, out, X+shiftx, Y+shifty);

        out(isnan(out)) = 0;
    end

    %Function to calculate the normalized joint histogram of the 2 images
    function out = jointHistogram(in1, in2)        
        is = size(I1);
        out = zeros(256);
        
        for i=1:is(1)
            for j=2:is(2)
                out(in1(i,j)+1, in2(i,j)+1) = out(in1(i,j)+1, in2(i,j)+1)+1;
            end
        end


    end

    %Function to calculate the marginal Entropy of one image
    function out = marginalEntropy(jh)
        marginal = sum(jh);
        out = 0;
        for i = 1:256;
            if(marginal(i) ~= 0)
                out = out - marginal(i)*log2(marginal(i));
            end
        end
    end
end
