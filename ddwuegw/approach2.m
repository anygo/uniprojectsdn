%% aufraeumen   
clear all;
close all;
clc;
pausetime = 0.01;
figure(1);
colormap gray;
    
%% bild laden    
img = rgb2gray(imread('BildSerie/b (9).jpg'));
img_org = img;
imagesc(img);
pause(pausetime);
img = imfilter(img, fspecial('gaussian', [25 25]));
imagesc(img);
pause(pausetime);
laplacian = imfilter(img, fspecial('laplacian'), 'replicate');
imagesc(laplacian);
pause(pausetime);

sigma = 2;
hsize = [5 5];
img2 = imfilter(img, fspecial('log', hsize, sigma));
imagesc(img2);
pause(pausetime);

%% thresholding
imgbw = im2bw(img2, graythresh(img2)); 
subplot(2,1,1), imagesc(img + uint8(~imgbw)*100), subplot(2,1,2), imagesc(imgbw);
drawnow;
pause(pausetime);

%% wildes morphing
imgbw = imclose(imgbw, strel('disk', 12));
subplot(2,1,1), imagesc(img + uint8(~imgbw)*100), subplot(2,1,2), imagesc(imgbw);
drawnow;
%pause(pausetime);

imgbw = imopen(imgbw, strel('disk', 50));
subplot(2,1,1), imagesc(img + uint8(~imgbw)*100), subplot(2,1,2), imagesc(imgbw);
drawnow;
%pause(pausetime);

imgbw = imclose(imgbw, strel('disk', 50, 0));
subplot(2,1,1), imagesc(img + uint8(~imgbw)*100), subplot(2,1,2), imagesc(imgbw);
drawnow;
%pause(pausetime);


%% kleine zellen finden 1
% segmented = img .* uint8(imgbw);
% figure(3);
% imagesc(histeq(segmented));
% title('krypten ausgeschnitten');
% colormap gray;

%% kleine zelle finden 2
%Ergebniss wird zZ nicht weiter verwende, obwohl es "human optical"
%vielversprechend aussieht
image_filtered_sharp = imfilter(img,fspecial('unsharp')); 

%groﬂer (25*25) Log mit kleinem Sigma
image_filtered = imfilter(img,fspecial('log', [25 25], 0.4));
%oder
%image_filtered = imfilter(adapthisteq(img_org),fspecial('log', 25, 0.4));

image_filtered(imgbw==0) = 0;

figure(4)
imagesc(image_filtered_sharp)
colormap gray
figure(5)
imagesc(image_filtered)
colormap gray






%% region growing
figure(5)
imagesc(imgbw);
colormap gray
[x_p,y_p] = ginput(1);



for i = 1:size(x_p,1)
I = im2double((img)); 
J = regiongrowing(I,round(x_p(i)),round(y_p(i)),0.1); 
figure(4)
imagesc(J)
end

