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
%        objDAPI
%
% Outputs:
%         core == image containing all expanded core
%
%         core_idx == cell array containing linear indices of cores

objDAPI = {s.objDAPI};

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
    for i = 1:length(objDAPI)
        curDAPI = objDAPI{i};
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