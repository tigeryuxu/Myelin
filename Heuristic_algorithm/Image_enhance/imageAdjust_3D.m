function [modImage, bw] = imageAdjust_3D(im, fillHoles, enhance)

%% Finds threshold to take out most of the noise
I = im;
% 
% %% Subtract background:
%% Subtract background:
I = imgaussfilt3(I, 2);
if enhance == 'Y'
    background = imopen(I,strel('disk',150));
    I2 = imsubtract(I, background);
    I = I2;
    %I = adapthisteq(I);
end
green_3D = imgaussfilt3(im,0.5);
bw = binarize_3D_otsu(im);
modImage = bw;
end