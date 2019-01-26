function [modImage, bw] = imageAdjust(im, fillHoles, enhance)

%% Finds threshold to take out most of the noise
I = im;
% 
% %% Subtract background:
%% Subtract background:
I = imgaussfilt(I, 2);
if enhance == 'Y'
    background = imopen(I,strel('disk',150));
    I2 = I - background;
    I = I2;
    %I = adapthisteq(I);
end

% if enhance == 'Y'    
%     I = adapthisteq(I);
%     test = adaptthresh(I, 0.6z);
% else
% end


%I = adapthisteq(I);

%% Then runs adjustments
level = graythresh(I);
if level == 0
    level = 1;
end

if level < 0.03
    level = 0.03;
end
    

bw = imbinarize(I, level);
%figure; imshow(bw);

modImage = imcomplement(bw);   % invert
modImage = bwareaopen(modImage, fillHoles);  % Eliminate holes
modImage = imcomplement(modImage);   % invert

bw = modImage;

end