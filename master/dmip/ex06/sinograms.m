function sinograms()
    close all;
    x1 = zeros(128,128);
    x1(20:30,20:30) = 1;
    x1(90:100,50:60) = 1;
    y1 = radon(x1);
    plot_sinogram(x1,y1);

    x2 = zeros(128,128);
    x2(60:68,60:68) = 1;
    y2 = radon(x2);
    plot_sinogram(x2,y2);

    x3 = zeros(128,128);
    [X Y] = meshgrid(1:128,1:128);
    x3((X-40).^2 + (Y-40).^2 < 25) = 1;
    x3(80:87,80:95) = 1;
    y3 = radon(x3);
    plot_sinogram(x3,y3);
end

function plot_sinogram(x, y)
    figure
    subplot(1,2,1);
    imagesc(x);
    axis image;
    subplot(1,2,2);
    imagesc(y);
    axis image;
    colormap gray;
end