%PM 2010
function detect_zellen()
close all 
image_t = imread('bild_sb/b (9).jpg');




%ränder wegschneiden falls nötig
%image_t = image_t(8:(size(image_t,1)-6),8:(size(image_t,2)-5),:); 1ver
image_t = image_t(3:(size(image_t,1)-2),3:(size(image_t,2)-2),:);

%keep color image
image_t_c = image_t;
%convert to grayscale
image_t = rgb2gray(image_t);

%hand-markieren von Bereichen die Becherzellen sind
 BW1 = roipoly(image_t);
 BW2 = roipoly(image_t);
 BW3 = roipoly(image_t);
 BW = BW1+BW2+BW3;
%alternative Maske laden, wenn immer dasselbe Bild bearbeitet wird, vorher
save('BWmask.mat','BW')
load('BWmask.mat');



%Detection von Krypten%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(1==1)
stepsize = [12 12];

% Bild in kleine bins stückeln / Größe: stepsize
% TO DO 2: Geht nur, wenn Höhe/Breite des Bildes teilbar durch stepsize
% image_t_cut = image_t(11:size(image_t,1),:); 1ver
image_t_cut = image_t;
B = splitarray(image_t_cut, stepsize);
bin_mat = zeros(size(B));

%Berechne Mittelwerte der bins
for i = 1:size(bin_mat,1)
    for j = 1:size(bin_mat,2)
        temp1 = mean(B{i,j});
    bin_mat(i,j) = mean(temp1(:));
    end
   
end

%Schwellwert auf bins anwenden und Ergebniss in neue Matrix verpacken
%krypmap(find(bin_mat>60))=255; %%%1 ver

thres = 40;
krypmap = zeros(size(bin_mat));
krypmap(find(bin_mat>thres))=255;


%find ajected regions / Zusammenhängende Regionen in Threshold

%Erweiter matrix unten und oben damit mein code zur ajected regions läuft -> to do 
%krypmap = [ones(size(krypmap,1),3)*255 krypmap ones(size(krypmap,1),3)*255];
%krypmap = [ones(3,size(krypmap,2))*255; krypmap; ones(3,size(krypmap,2))*255];

%run ajected regions code
[labelmap] = morphopperator(krypmap,image_t_c);

%eliminated small regions
for i = 1:max(labelmap(:))
    if(size(labelmap(find(labelmap==i)),1)<25)
        labelmap(find(labelmap==i)) = 0;
    end
end

%dilate image  
dilatemap = ones(size(labelmap))*255;
dilatemap(find(labelmap~=0))=0;
[dil_image] = dilate_img(dilatemap,[5 5]);

figure(1)
imagesc(dil_image)
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%Verhältniss Becherzellen / Epithelzellen%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%blurring entfernen
if(1==1)
lap_filter = fspecial('sobel');
image_filtered = conv2(image_t,lap_filter);
imagesc(image_filtered)

image_filtered = image_filtered(13:size(image_filtered,1),3:size(image_filtered,2));
B = splitarray(image_filtered, stepsize);
bin_mat_filt = zeros(size(B));

for i = 1:76
    for j = 1:77
        temp1 = mean(B{i,j});
    bin_mat_filt(i,j) = max(temp1(:));
    end
   
end
end
%TO DO 1: bild wird gefiltert aber Ergebniss nicht angewendet ->lbp
%anwenden?
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%bin_mat_filt(find(dil_image(3:(size(dil_image,1))-4,3:(size(dil_image,2))-4)~=255))=0;

%Compute mean value of Becherzelle aus hand-Eingabe (BW)
mean_BW = round(mean(image_t(find(BW~=0))));

%Copy image for coloring
image_t_c_bech = image_t_c;

%creat matrix containing 1=becherzelle / 0=else
bandpass = 15;
becher_position = uint8(zeros(size(image_t)));
%Schwellwertanwendung im bereich: (mean_BW+bandpass) > becher_position < (mean_BW+bandpass)
becher_position(find(image_t>mean_BW-bandpass & image_t<mean_BW+bandpass))=1;


%ajected region search
%find ajected regions / eliminate small regions
if(1==1)

    %lable according to search algo
becher_position(find(becher_position==0))=255;
becher_position(find(becher_position==1))=0;

  %resize aroding to search algo -> weil mein code das nicht kann :-) ->TO
  %DO
becher_position = [ones(size(becher_position,1),3)*255 becher_position ones(size(becher_position,1),3)*255];
becher_position = [ones(3,size(becher_position,2))*255; becher_position; ones(3,size(becher_position,2))*255];

[labelmap] = morphopperator(becher_position,image_t_c);

%eliminate small regions
for i = 1:max(labelmap(:))
    if((size(find(labelmap==i),1))<80)
       labelmap(find(labelmap==i))=0; 
    end  
end


%resize back to original size 
labelmap(1:3,:)=[];
labelmap(size(labelmap,1)-2:size(labelmap,1),:)=[];
labelmap(:,1:3)=[];
labelmap(:,size(labelmap,2)-2:size(labelmap,2))=[];

%resize back to original size 
becher_position(1:3,:)=[];
becher_position(size(becher_position,1)-2:size(becher_position,1),:)=[];
becher_position(:,1:3)=[];
becher_position(:,size(becher_position,2)-2:size(becher_position,2))=[];

becher_position(find(labelmap==0))=0;
becher_position(find(labelmap~=0))=255;

end
figure(1)
imagesc(becher_position)

se = strel('disk',1);
becher_position = imdilate(becher_position,se);
figure(2)
imagesc(becher_position)

image_t_c_bech(find(becher_position==255))=255;
%image_t_c_bech(find(image_t>mean_BW-bandpass & image_t<mean_BW+bandpass))=255;




%resize matrix containing Krypten positions back to org. image size
dil_imag_org_size = (imresize(dil_image(3:(size(dil_image,1))-3,3:(size(dil_image,2))-4),12));
dil_imag_org_size = dil_imag_org_size(1:(size(dil_imag_org_size,1))-2,:);

%color areas with Krypten in black
image_t_c_bech(find(dil_imag_org_size==0))=0;

%Compute ratio of becherzellen
size_becher = size(find(becher_position==255),1);
size_krypten = size(find(dil_imag_org_size==0),1);
size_gesamt = size(image_t,1)*size(image_t,2);
all_relevant = size_gesamt-size_krypten;
size_becher/all_relevant*100

figure(3)
imagesc(image_t_c_bech);
end

function [labelmap] = morphopperator(filter_resp, orig_image)


%dilate image
dil_factor = ([1 1]);
dil_image = dilate_img(filter_resp, dil_factor);


labelmap = zeros(size(filter_resp,1),size(filter_resp,2));
labelcount = 1;
neighbor_vec = [];
original_filter_resp=filter_resp;
filter_resp = dil_image;
%find ajected reagions
for i = 2:size(filter_resp,1)-1
    for j = 2:size(filter_resp,1)-1
        if(filter_resp(i,j)==0)
            %pixel is part of goupe "labelcount"
            labelmap(i,j)=labelcount;
            %set pixel to "zero" (non-sick group)
            filter_resp(i,j) = 255;
            %push pixelposition in vector containing pixels with potential
            %neighbors
            neighbor_vec = [neighbor_vec; [i j]];
            
            %check if their are pixels, where neighbors have not been
            %controlled for conntected pixels
            while(size(neighbor_vec,1)~=0)
                %search for neighbors
                for k = [-1 0 0 1]
                    for l = [0 -1 1 0]
                        if(filter_resp(neighbor_vec(1,1)+k,neighbor_vec(1,2)+l)==0)
                            %pixel is part of goupe "labelcount"
                            labelmap(neighbor_vec(1,1)+k,neighbor_vec(1,2)+l)=labelcount;
                            %push pixelposition in vector containing pixels with potential
                            %neighbors
                            neighbor_vec = [neighbor_vec; [neighbor_vec(1,1)+k neighbor_vec(1,2)+l]];
                            %set pixel to "zero" (non-sick group)
                            filter_resp(neighbor_vec(1,1)+k,neighbor_vec(1,2)+l) = 255;
                        end
                    end
                end
                neighbor_vec(1,:) = [];
            end
            labelcount = labelcount +1;
        end
        
    end
    
end

%h = waitbar(0,'Please wait...');
% orgdata=orig_image;
% labelmap = imresize(labelmap, 12);
% %plot data
% for i = 1:max(labelmap(:))    
%     [o] = find(labelmap==i);
%     [x, y] = find(labelmap==i);
%     
%      waitbar(i/max(labelmap(:)))
%     %controll, if they are border pixels
%     for n = 1:size(x,1)
%        
%             for k = [-1 0 0 1]
%                 for l = [0 -1 1 0]
%                     if(labelmap(x(n)+k,y(n)+l)~=0)
%                         orig_image(x(n),y(n),1) = 255;
%                     end
%                 end
%             end
%       
%     end
%    
%     %orig_image(o)=255; 
% end
% 
% dummy = 1;
%  close(h) 
% figure(1)
% subplot(1,3,1)
% imagesc(orig_image)
% axis square
% subplot(1,3,2)
% hold on
% imagesc(original_filter_resp)
% colormap gray
% axis square
% hold off
% 
% subplot(1,3,3)
% hold on
% imagesc(orgdata)
% axis square
% hold off

end

function [dil_image] = dilate_img(image_in, dil_factor)



%dilate image
dilate_result = ones(size(image_in))*255;
dilate_factor = dil_factor;
struc_element = ones(dilate_factor(1),dilate_factor(2))*255;

for i = 1:size(image_in,1)-size(struc_element,1)+1
    for j = 1:size(image_in,2)-size(struc_element,2)+1
        relevant_sec = image_in(i:i+size(struc_element,1)-1,j:j+size(struc_element,2)-1);

        if(sum(sum(relevant_sec-struc_element))<0)
        dilate_result(i+floor(size(struc_element,1)/2),j+floor(size(struc_element,2)/2))= 0;
        end
    end
end

%add dilate result to original threshold image
image_in(find(dilate_result==0))=0;
dil_image = image_in;
% imagesc(image_in)
% colormap gray

end

function B = splitarray(A, siz)
%SPLITARRAY Split an array into subarrays.
%
%   SPLITARRAY(A, SIZ) splits the array A into subarrays of size SIZ and
%   returns a cell array with all the subarrays.
%
%   See also NUM2CELL, MERGEARRAY.

%   Author:      Peter John Acklam
%   Time-stamp:  2002-03-03 13:50:35 +0100
%   E-mail:      pjacklam@online.no
%   URL:         http://home.online.no/~pjacklam

error(nargchk(2, 2, nargin));
if length(siz) == 1, siz = [siz siz]; end

Asiz = size(A);                     % size of A
Adim = length(Asiz);                % dimensions in A
Sdim = length(siz);                 % dimensions in each subarray
Bdim = max(Adim, Sdim);             % dimensions in B (output array)
Asiz = [Asiz ones(1,Bdim-Adim)];    % size of A (padded)
Ssiz = [siz  ones(1,Bdim-Sdim)];    % size of each subarray (padded)

Bsiz = Asiz ./ Ssiz;                % size of B (output array)
if any(Bsiz ~= round(Bsiz))
   error('A can not be divided into subarrays as specified.');
end

% A becomes [ Ssiz(1) Asiz(1)/Ssiz(1) Ssiz(2) Asiz(2)/Ssiz(2) ... ].
A = reshape(A, reshape([Ssiz ; Bsiz], [1 2*Bdim]));

% A becomes [ Ssiz(1) Ssiz(2) ... Asiz(1)/Ssiz(1) Asiz(2)/Ssiz(2) ... ].
A = permute(A, [ 1:2:2*Bdim-1 2:2:2*Bdim ]);

% A becomes [prod(Ssiz) prod(Bsiz)].
A = reshape(A, [prod(Ssiz) prod(Bsiz)]);

B = cell(Bsiz);
for i = 1:prod(Bsiz)
   B{i} = reshape(A(:,i), Ssiz);
end
end


%region growing%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure(1)
% imagesc(image_t);
% colormap gray
% [x_p,y_p] = ginput(1);
% 
% 
% 
% for i = 1:size(x_p,1)
% I = im2double((image_t)); 
% J = regiongrowing(I,round(x_p(i)),round(y_p(i)),0.2); 
% figure(2)
% imagesc(J)
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

