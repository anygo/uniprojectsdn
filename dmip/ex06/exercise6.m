%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Diagnostic Medical Image Processing (DMIP) 
% WS 2010/11
% Exercise: Filtered Backprojection for parallel beam
% NOTE: Complete the '???' lines
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function fbp()
    clear all;
    close all;
    close all hidden;
    clc;

    im = phantom(64);

    % Size of the input image
    [m,n] = size(im);
    
    im = mat2gray(im);
    
    figure(8);
    subplot(1,2,1); imagesc(im); colormap gray; axis image; xlabel('x'); ylabel('y'); title('Original phantom image');
    
    disp('Extracting projections ...'); 

    % Acquire the projection sequence
    % Compute "numberOfProjections" 1-D projections along angle given the "startAngle" and the "angleIncrement". 
    % The result is a parallel projection of the image.
    
    %angleIncrement = ???; 
    %startAngle = ???;
    phi = startAngle;
    %numberOfProjections = ???;

    % Create a cell array for projections 
    projs = cell(numberOfProjections, 1);

    for i=1:numberOfProjections
        % Save the angles of the numberOfProjections in an array
        phis(i) = phi;

        % Image is rotated by phi degrees counterclockwise, so the projection is lined up in the
        % columns & finally the columns are summed up
        %rI = ???

        figure(1);
        imagesc(rI);
        xlabel('x');
        ylabel('y');
        axis image
        colormap gray;
        title(['Angle: ', num2str(phi), '°']);
        drawnow;

        % Sum up columnwise -> parallel beam
        %projs{i} = ???

        % Compute the next rotation angle
        %phi = ??? 
    end
    
    fltr = 0; % No filtering, no high-pass filter is applied!
    % fltr = 1; % Ram-Lak
    % fltr = 2; % Shepp-Logan

    % Reconstructed image
    ct = zeros(size(im));  

    % Half of the dimension
    dimY2 = m/2; 
    dimX2 = n/2;

    for phi = 1:numberOfProjections  
        disp('Processing projection angle: ')
        disp(phi)

        proj = projs{phi};
        ang = phis(phi);
        dimensions = [length(proj) length(proj)];

        % [-m/2,  -n/2]
        Po = [-dimensions(1)/2; -dimensions(2)/2];

        % [m/2, -n/2]
        Pt = [dimensions(1)/2; -dimensions(2)/2];

        % Which filter should be used?
        if(fltr == 1)
            %fltm = ???
            % Compute the filter step
            %proj = ???
        elseif(fltr == 2)
            %fltm = ???
            % Compute the filter step
            %proj = ???
        else
            % No filter is applied
            proj = proj;
        end

        FBP = zeros(m,n); 

        % Always sample in the space where you expect the result!
        for j = 1:m     
            for i = 1:n
                % Compute with the help of the index position of the projected pixel (i,j)
                % in the filtered projection image the detector position t.
                t = footDist( [(i-1)-(dimX2),((j-1))-(dimY2)], Po, Pt, ang );           
                
                % Compute the left and right pixel (integer position)
                % position of computed index t.
                % Perform a dimension check if the index t is inside the detector row.
                lowerIndex = max(1, min(dimensions(2), 1.0*(floor(t))));
                upperIndex = min(dimensions(2), lowerIndex+1);
                
                % Distance to the computed left and right pixel neighbors on the detector
                lowerDiff = norm(t-lowerIndex);
                upperDiff = norm(t-upperIndex);

                % Linear interpolation
                FBP(j,i) = (1-lowerDiff)*proj(lowerIndex) + (1-upperDiff)*proj(upperIndex);
                
                % sprintf('(i,j)=(%i,%i) -> t=%f [lI=%f,uI=%f]=%f', i, j, t, lowerIndex, upperIndex, FBP(j,i))
                
%                 figure(7);
%                 imagesc(FBP); axis image; colormap gray; xlabel('x'); ylabel('y'); title('FBP');
%                 drawnow;
            end % i
       end % j
       % Accumulate FBPs
       %ct = ???
       
       figure(8);
       subplot(1,2,2); imagesc(ct); axis image; colormap gray; xlabel('x'); ylabel('y'); title(['Projection: ', num2str(phi), '/', num2str(numberOfProjections)]);
       drawnow;
       hold off
    end % loop projections

    % Normalizing according to projections
    ct = ct/numberOfProjections; 
    
    figure(8);
    subplot(1,2,2); imagesc(ct); axis image; colormap gray; xlabel('x'); ylabel('y'); title('Reconstruction result');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [shepp] = SheppLogan(width)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [shepp] = SheppLogan(width)
% Shepp-Logan filtering with kernel size "width" is applied.
% Formula: chapter "Reconstruction", slide: 47/63


% ???


end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [ramlak] = RamLak(width)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ramlak] = RamLak(width)
% Ram-Lak filtering with kernel size "width" is applied.
% Formula: chapter "Reconstruction", slide: 48/63


% ???


end
