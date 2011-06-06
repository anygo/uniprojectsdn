function filtered_back_projection()
    clear all;
    close all;
    close all hidden;
    clc;

    % test image
    im = phantom(64);

    % size of the input image
    [m, n] = size(im);

    figure(8);
    subplot(1,2,1); imagesc(im); colormap gray; axis image;
    title('Original phantom image');

    disp('Extracting projections ...');

    angleIncrement = 1;
    startAngle = 0;
    phi = startAngle;
    numberOfProjections = ceil(180/angleIncrement); % we need 180 degrees

    fltr_size = 15;
    %fltm = [1]; % no filter
    %fltm = RamLak(fltr_size);
    fltm = SheppLogan(fltr_size);


    % cell for storing the projections
    projs = cell(numberOfProjections, 1);
    phis = zeros(numberOfProjections, 1);

    for i = 1:numberOfProjections
        % save the angles of the numberOfProjections in an array
        phis(i) = phi;

        % image is rotated by phi degrees counterclockwise, so the 
        % projection is lined up in the columns and finally the columns are
        % summed up
        rI = imrotate(im, phi, 'bilinear');

        % sum up columnwise -> parallel beam
        projs{i} = sum(rI);

        % compute the next rotation angle
        phi = phi + angleIncrement;
    end

    plot_sinogram(projs);

    % reconstructed image
    ct = zeros(size(im));

    % half of the dimension
    dimY2 = m/2;
    dimX2 = n/2;

    for phi = 1:numberOfProjections
        disp(['Processing projection angle: ' num2str(phi)]);

        proj = projs{phi};
        ang = phis(phi);
        dimensions = [length(proj) length(proj)];

        % [-m/2,  -n/2]
        Po = [-dimensions(1)/2; -dimensions(2)/2];

        % [m/2, -n/2]
        Pt = [dimensions(1)/2; -dimensions(2)/2];

        % filter
        proj = conv(proj, fltm, 'same');

        % init image for intermediate step
        FBP = zeros(m, n);

        % sample image
        for j = 1:m
            for i = 1:n
                % compute with the help of the index position of the projected pixel (i,j)
                % in the filtered projection image the detector position t.
                t = footDist( [(i-1)-(dimX2),((j-1))-(dimY2)], Po, Pt, ang );

                % compute the left and right pixel (integer position)
                % position of computed index t.
                % perform a dimension check if the index t is inside the detector row.
                lowerIndex = max(1, min(dimensions(2), 1.0*(floor(t))));
                upperIndex = min(dimensions(2), lowerIndex+1);

                % distance to the computed left and right pixel neighbors on the detector
                lowerDiff = norm(t-lowerIndex);
                upperDiff = norm(t-upperIndex);

                % linear interpolation
                FBP(j,i) = (1-lowerDiff)*proj(lowerIndex) + (1-upperDiff)*proj(upperIndex);

            end
        end
        
        % accumulate FBPs
        ct = ct + FBP;

        % plot current stage
        figure(8);
        subplot(1,2,2); imagesc(ct); axis image; colormap gray;
        title(['Projection: ', num2str(phi), '/', num2str(numberOfProjections)]);
        drawnow;
        hold off
    end 

    % normalize according to #projections
    ct = ct/numberOfProjections;

    figure(8);
    subplot(1,2,2); imagesc(ct); axis image; colormap gray; 
    title('Reconstruction result');
end

function [shepp] = SheppLogan(width)
    % compute Shepp-Logan filter kernel
    % formula taken from chapter "Reconstruction"

    shepp = zeros(1, width);
    for t = -floor(width/2):floor(width/2)
        val = -2 / (pi *(4*t^2 - 1));

        shepp(1, 1 + t + floor(width/2)) = val;
    end

end

function [ramlak] = RamLak(width)
    % compute RamLak filter kernel
    % formula taken from chapter "Reconstruction"

    ramlak = zeros(1, width);
    for t = -floor(width/2):floor(width/2)
        val = 0;
        if t == 0
            val = pi/4;
        end
        if mod(t, 2) == 0
            val = 0;
        end
        if mod(t, 2) == 1
            val = -1 / (pi*t^2);
        end
        ramlak(1, 1 + t + floor(width/2)) = val;
    end

end

function plot_sinogram(p)
    % plots a sinogram
    
    max_size = 0;
    for i = 1:size(p, 1);
        if size(p{i}, 2) > max_size
            max_size = size(p{i}, 2);
        end
    end

    sinogram = zeros(size(p,1), max_size);

    for i = 1:size(p,1)
        from = floor((max_size - size(p{i},2))/2)+1;
        sinogram(i, from:from+size(p{i}, 2) - 1) = p{i};
    end

    figure;
    imagesc(sinogram');
    title('sinogram');
    xlabel('projection');
    colormap gray;
end
