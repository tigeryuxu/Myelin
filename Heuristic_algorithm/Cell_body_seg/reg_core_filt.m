function [s] = reg_core_filt(bw, diameter, siz, percent_Dilate, s)

% Regional core filter:
% (1) Finds IMEXTENDEDMAX areas of the BWDIST transform
% (2) Filters out regions with values < DIAMETER accepted
% (3) Expands each remaining region until hits nearby DAPI || exceeds MAX SIZE
%
% Inputs:
%        bw == bw image of O4 stain
%        diameter
%        max size
%        %
% Outputs:
%        updates s - struct

objDAPI = {s.objDAPI};
DAPI_im = zeros(siz);
for i = 1:length(objDAPI)
    DAPI_im(objDAPI{i}) = 1;
end

A = bw;
A = imcomplement(A);   % invert

%% Algorithm
D = bwdist(A);  % Euclidean distance

%% Take out holes depending on regional-maxima
tmp = D;
regionalTmp = imextendedmax(tmp, 2);
regionalCore = D;

% Filter out all regional maxima that are smaller than "diameter"
idxReg = find(regionalTmp);
idxThres = find(regionalCore > diameter);
same = ismember(idxReg, idxThres);  % find the identical value
if ~isempty(find(same, 1))
    regionalTmp(idxReg(~same)) = 0;
end

%% Search AROUND the regional core to a distance of current core/percent_Dilate
div = bwconncomp(regionalTmp);
for t = 1:length(div.PixelIdxList)
    curCore = div.PixelIdxList{t};
    
    tmp = zeros(siz);
    val_reg = D(div.PixelIdxList{t});
    tmp(curCore) = 1;
    
    core_dist = bwdist(tmp);   % matrix that is distance to the object in "tmp"
    
    allMin = [];
    allIdx = [];
    %% Loop through all the objDAPI to find one closest
    %% ADDED EXTRA PROCESSING to speed up - Tiger Xu 06/03/2019
    % but first eliminate furthest DAPI by expanding the "curCore" - tmp
    % image to eliminate anything outside of a certain distance away
    
%     [x_size, y_size] = ind2sub(siz, curCore(1));
%     width = siz(1);
%     height = siz(2);
%     
%     if width > 3000 || height > 3000  % for larger images, do faster searching
%        [mask] = create_mask(x_size, y_size, width, height, siz);
%         
%         %dil_tmp = imdilate(tmp, strel('disk', 100));
%         masked_DAPI = DAPI_im;
%         masked_DAPI(~mask) = 0;
%     else
        masked_DAPI = DAPI_im;
%    end
    
    cc = bwconncomp(masked_DAPI);
    for i = 1:length(cc.PixelIdxList)
        curDAPI = cc.PixelIdxList{i};
        allMin = [allMin min(core_dist(curDAPI))];
        allIdx = [allIdx i];
    end
    
    % Add the "core" to the cell with minimum distance:
    [val, minIdx] = min(allMin);
    cellIdx = allIdx(minIdx);
    
    % that is also closer than the threshold of the core
    max_thresh =  floor(mean(double(val_reg))/percent_Dilate);
    
    if val < max_thresh
        s(cellIdx).Core = curCore;
    end
end

end



function [mask] = create_mask(x_size, y_size, width, height, siz)

        total_length_x = 400;
        total_length_y = 400;
        
        x_left = x_size - total_length_x / 2;
        x_right = x_size + total_length_x / 2;
        
        % adaptive cropping for width (x-axis)
        if x_left <= 0
            x_right = x_right + abs(x_left) + 1;
            x_left = 1;
            
        elseif x_right > width
            x_left = x_left - (x_right - width);
            x_right = width;
        end
        
        % adaptive cropping for height (y-axis)
        y_top = y_size - total_length_y / 2;
        y_bottom = y_size + total_length_y / 2;
        if y_top <= 0
            y_bottom = y_bottom + abs(y_top) + 1;
            y_top = 1;
            
        elseif y_bottom > height
            y_top = y_top - (y_bottom - height);
            y_bottom = height;
        end
        
        % Final check to see if sizes are correct
        if (x_right - x_left) ~= 100 || (y_bottom - y_top) ~= 100
            %break;
            j = 'ERROR in crop size';
        end
        
        mask = zeros();
        mask(x_left:x_right, y_top:y_bottom) = 1;
end