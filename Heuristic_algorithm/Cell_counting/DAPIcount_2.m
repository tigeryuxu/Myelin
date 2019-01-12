function [mat, objDAPI, bw] = DAPIcount_2(intensityValueDAPI, DAPIsize, DAPImetric, enhance, DAPI_bb_size)

% Cell DAPI nuclei counter by performing standard binarization and watershed operations:
% - removes high intensity pixels (> 0.15) to get better binarization
% - "optional" enhacement ==> imadjust
% - edge exclusion (no DAPI along edges of image
% - watershed (optional imimposemin)
% - roundness exclusion criteria
% 
% Also includes:
% - artifiact removal
% 
% Inputs:
% 
% 
% Outputs:
%             mat == matrix of centroids of identified DAPI objects
%             objDAPI == PixelIdxList of all identified DAPI objects
%             bw == logical image of DAPI objects

%% Smooth
intensityValueDAPI = imgaussfilt(intensityValueDAPI, 2);

%% Threshold
I = intensityValueDAPI;

%% Subtract background:
if enhance == 'Y'
    background = imopen(I,strel('disk',DAPI_bb_size));
    I2 = I - background;
    I = I2;
    I = adapthisteq(I);
end

% %% To deal with artifacts: (remove???)
tmpArt = imbinarize(I);
%figure; imshow(tmpArt);
tmpArt = bwconncomp(tmpArt);

idxArt = [];
i = 1;
while i < length(tmpArt.PixelIdxList)
    if length(tmpArt.PixelIdxList{i}) > 150000
        idxArt = [idxArt; tmpArt.PixelIdxList{i}];
        I(idxArt) = 0;
        tmpArt = imbinarize(I);
        tmpArt = imdilate(tmpArt, strel('disk', 5));
        %figure; imshow(tmpArt);
        tmpTmp = tmpArt;
        tmpArt = bwconncomp(tmpArt);
        i = 1;
        
    else
        i = i+1;
    end
    if length(tmpArt.PixelIdxList{i}) < DAPIsize
        idxArt = [idxArt; tmpArt.PixelIdxList{i}];
        i = i+1;
    end
end
I(idxArt) = 0;
%figure; imshow(imbinarize(I));


%% Binarize:
% if enhance == 'Y'
%     
% else
%     I(I > 0.15) = 0.15;    % ELIMINATE SUPER HIGH INTENSITY STUFF
% end
bw = imbinarize(I);

%% Subtract out the edges of the image to eliminate edge DAPI
% thick = 10;
% tmp = zeros(siz);
% tmp(1:thick, :) = 1;
% tmp(siz(1) - thick : siz(1), :) = 1;
% tmp(:, 1:thick) = 1;
% tmp(:, siz(2) - thick: siz(2)) = 1;
% bw = bw - tmp;
% bw(bw < 0) = 0;
% %figure(102); imshow(bw);

%% Find min mask
bw = ~bwareaopen(~bw, 10);  % clean
D = -bwdist(~bw);  % EDT
mask = imextendedmin(D, 2);   % Extended minima


%% Watershed segmentation by imposing minima (NO NEED FOR imposing minima) probably b/c gaussfilt renders this useless
D2 = imimposemin(D, mask);

Ld2 = watershed(D2);
bw3 = bw;
bw3(Ld2 == 0) = 0;
bw = bw3;

%figure(100); title('DAPI watershed');
[B,L] = bwboundaries(bw, 'noholes');

%imshow(bw);
%imshow(label2rgb(L, @jet, [.5 .5 .5]));
%hold on;

%% Roundness metric
stats = regionprops(L,'Area','Centroid', 'PixelIdxList'); %%%***good way to get info about region!!!

thresholdDAPI = DAPImetric;
objDAPI = cell(1); idx = 1; mat = [];
for k = 1:length(B)
    boundary = B{k};                                             % obtain (X,Y) boundary coordinates corresponding to label 'k'
    delta_sq = diff(boundary).^2;                           % compute a simple estimate of the object's perimeter
    perimeter = sum(sqrt(sum(delta_sq,2)));
    area = stats(k).Area;                                       % obtain the area calculation corresponding to label 'k'
    metric = 4*pi*area/perimeter^2;                      % compute roundness metric
    metric_string = sprintf('%2.2f',metric);           % display
    if metric >= thresholdDAPI   && (area > DAPIsize && area < 1000000)  %and within size range
        centroid = stats(k).Centroid;
        %plot(centroid(1),centroid(2),'k.');
        mat = [mat ; [centroid(1) centroid(2)]];       % add to matrix of indexes
        objDAPI{idx} = stats(k).PixelIdxList;
        idx = idx + 1;
        %text(boundary(1,2)-35,boundary(1,1)+13,metric_string,'Color','y', 'FontSize',8);
        
    else
        bw(stats(k).PixelIdxList) = 0;    % Otherwise, set to 0 in binary mask
    end
end
%hold off;
end

