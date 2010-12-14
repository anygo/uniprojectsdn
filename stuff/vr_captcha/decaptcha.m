clear all;
clc;

% decaptcha the vr-crap-captcha
n_images = 3;
for n = 1:n_images
    img = imread([num2str(n) '.jpg']);
    img = rgb2gray(img);
    img = im2bw(img, graythresh(img));
    img = ~img;
    img = img(2:end-1, 2:end-3);
    subplot(n_images,2,(n-1)*2+1), imagesc(img);
    
    horizontal_lines = [];
    vertical_lines = [];
    
    for i = 1:size(img,1)
        if all(img(i,:))
            horizontal_lines = [horizontal_lines; i];
        end
    end
    
    for i = 1:size(img,2)
        if all(img(:,i))
            vertical_lines = [vertical_lines; i];
        end
    end

    
    img(horizontal_lines,:) = 0;
    img(:,vertical_lines) = 0;
    subplot(n_images,2,(n-1)*2+2), imagesc(img);
    
%     img = imdilate(img, strel('line', 1, 90));
%     subplot(n_images,2,(n-1)*2+2), imagesc(img);
%     
%     img = imclose(img, strel('line', 2, 60));
%     subplot(n_images,2,(n-1)*2+2), imagesc(img);
%     
%     img = imclose(img, strel('line', 2, -45));
%     subplot(n_images,2,(n-1)*2+2), imagesc(img);
%     
%     img = imclose(img, strel('line', 2, 90));
%     subplot(n_images,2,(n-1)*2+2), imagesc(img);
    

    
    %         img = imclose(img, strel('rect', [2 2]));
    %         subplot(n_images,2,(n-1)*2+2), imagesc(img);
    
    % scan image for different chars
%     img = uint8(img);
%     scanline = 1;
%     curVal = 1;
%     while true
%         if scanline == size(img, 2)
%             break;
%         end;
%         if ~any(img(:,scanline))
%             scanline = scanline+1;
%         else
%             while true
%                 if sum(img(:,scanline:scanline+3)) > 0
%                     ausschnitt = img(:,scanline:scanline+3);
%                     ausschnitt(ausschnitt == 1) = curVal;
%                     img(:,scanline:scanline+3) = ausschnitt;
%                     scanline = scanline+1;
%                     if scanline == size(img, 2)-2
%                         break;
%                     end;
%                 else
%                     scanline = scanline+1;
%                     break;
%                 end
%             end
%             curVal = curVal + 1;
%         end
%     end
    
    imagesc(img);
    drawnow;
    
end
