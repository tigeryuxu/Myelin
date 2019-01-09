function [s] = small_del_O4(O4_im, minLength, squareDist, siz, s)

% Finds the O4_im associated with counted O4
% then looks at these objects to determine if they are too small or not:
% (1) must have bounding box that is > 200 in BOTH height AND width
% (2) if one or the other is > 200, but the other is NOT, then check to see if that one is > 100 or not
%     if not > 100, then still doesn't count (i.e. the object is basically a straight line)
   
comp = bwconncomp(O4_im);
ass_O4 = zeros(siz);

match_objDAPI_idx = cell(0);
for Y = 1:length(comp.PixelIdxList)
    curObj = comp.PixelIdxList{Y};
    for Z = 1:length(s)
        curO4 = s(Z).Core;
        same = ismember(curObj, curO4);
        if ~(isempty(find(same, 1)))
            ass_O4(comp.PixelIdxList{Y}) = 1;   % also create pic of the ASSOCIATED objects too
            match_objDAPI_idx{end + 1} = Z;
            break;
        end
    end
end       

%% Take the associated image, and create boxes to subtract out morphologically small stuff:
ass_O4 = bwareaopen(ass_O4, 500);

stats = regionprops(ass_O4, 'BoundingBox', 'Extent', 'Centroid', 'PixelIdxList');

for i = 1:length(match_objDAPI_idx)
    idx_s = match_objDAPI_idx{i};
    cur_Core = s(idx_s).Core;

    % First find object corresponding to "core" in "stats"
    idx_match = 0;
    for j = 1:length({stats.PixelIdxList})
        same = ismember(stats(j).PixelIdxList, cur_Core);
        if ~isempty(find(same,1))
            idx_match = j;   % should only be 1 object anyways
            break;
        end
    end
   
    if idx_match > 0
        height = stats(idx_match).BoundingBox(4);
        width = stats(idx_match).BoundingBox(3);
        
        if (height < squareDist) && (width < squareDist/2)   % TOO SMALL
            s(idx_s).Bool_W = -1;   % -1 is permanent not wrapped
            stats(idx_match).Centroid = [];
        
        %% REMOVED FOR HUMAN OL analysis
        %elseif ((height > squareDist) && (width < squareDist/2)) || ((height < squareDist/2) && (width > squareDist)) % TOO SMALL
        %    s(idx_s).Bool_W = -1;
        %    stats(idx_match).Centroid = [];
        end
    end
end

end