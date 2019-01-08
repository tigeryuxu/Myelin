function [cores,  cb,  cb_O4, s] = O4_count(O4_im, cores, cb, siz, diameterFiber, density, s)

%% ***USER PICKED, maybe different for low vs. high density images???
if density == 'N'
    clean_s = 200; % for LOW DENSITY
elseif density == 'Y'
    clean_s = 1000; % for HIGH DENSITY???
end

%% Pre-process:
% by closing image (i.e. try to associate nearby things)
O4_im = imdilate(O4_im, strel('disk', 5));

%% unassociated cores:
% take the "cores" and match it to objects from O4_im
% all objects that match are excluded, as well as objects < certain size
comp = bwconncomp(O4_im);
unass_O4 = zeros(siz);
for Y = 1:length(comp.PixelIdxList)
    curObj = comp.PixelIdxList{Y};
    for Z = 1:length(s)
        curCB = s(Z).CB;
        same = ismember(curObj, curCB);
        if ~(isempty(find(same, 1))) || length(curObj) < clean_s
            comp.PixelIdxList{Y} = [];
            break;
        end
    end
    if ~isempty(comp.PixelIdxList{Y})
        unass_O4(comp.PixelIdxList{Y}) = 1;
    end
end

%% must also subtract fibers from this image, b/c don't want to consider those as unassociated cores
%% or maybe do this O4_count AFTER, associating all the fibers already??? so only get the left-over fibers

cb_O4 = unass_O4;
obj_O4 = bwconncomp(cb_O4);
cb_O4_idx  = obj_O4.PixelIdxList;


%% Find unassociated objects that match with DAPI, and add it to "Core"
%***only if the DAPI does NOT already have a core associated with it
cores_O4 = zeros(siz);
idx_new = cell(0);
for Y = 1:length(cb_O4_idx)
    cur_core = cb_O4_idx{Y};
    if isempty(cur_core)   %% SPEED UP CODE BY REDUCING REDUNDANCY
        continue;
    end
    for T = 1:length({s.objDAPI})
            DAPI = s(T).objDAPI;
            same = ismember(DAPI, cur_core);
            if ~isempty(find(same, 1))  &&  isempty(s(T).Core);
                overlap_idx = find(same);
                overlap = DAPI(overlap_idx);
                cores_O4(overlap) = 1;
                s(T).Core = overlap;   % ADDS TO "s" and then BREAKS ==> so ONLY ADDS 1 O4
                idx_new{end + 1} = T;
                break;
            end
    end
end

%% ***REMEMBER, at this point, associating "CB" with a cell is NOT counting it as O4+
%only when you do match_cores and add the "DAPI" as a "Core" is is marked as O4+

%% Run cell_body_filt to get segmented CB's to add to "cb", so later when sub_fibers, dont't sub huge blob 
%it's okay that cell_body_filt adds to "s" struct as well, b/c should be same still, and decide later what is O4+

%[cb_new, no_dilate_cb, s] = cell_body_filt(DAPI_cb_O4, diameterFiber, lengthY, lengthX, s);

%% USE NORMAL CB???

A = O4_im;
A = imcomplement(A);   % invert
D = bwdist(A);  % Euclidean distance
tmp = D;
tmp(tmp > diameterFiber) = 0;    % Take out Euclidean holes
core_tmp = D - tmp;  % subtract out the background to get the core
core_tmp(core_tmp < 0) = 0;
cb_new = bwmorph(core_tmp, 'thicken', diameterFiber);  % Then thicken the edges back to the original ledge

obj = bwconncomp(cb_new);
cb_idx = obj.PixelIdxList;

for j = 1:length(cb_idx)
    cur_cb = cb_idx{j};
    
    match = 0;
    for i = 1:length(idx_new)
        idx_s = idx_new{i};
        
        % ONLY IF THERE IS ALREADY A MATCHING CORE, WE SAVE CB
        if ~isempty(s(idx_s).Core)
            curCore = s(idx_s).Core;
            same = ismember(cur_cb, curCore);
            if ~isempty(find(same, 1))
                s(idx_s).CB = cur_cb;
                match = 1;
            end
        end
    end
%     if no match found, then delete the CB so can never be considered "wrapped"
%     ***BUT, can still be considered O4+
    if match == 0
        cb_new(cur_cb) = 0;  % sets CB to be zero
        % s(idx_s).CB = s(idx_s).Core;
    end
    
end


%% Add the 2 cb's together to get ALL CB's

%cb_new = imbinarize(cb_new + DAPI_cb_O4);

%% Match_cores helps ensure only 1-ish core per CB
%overlap_thresh = 0;
%[cores_m,cb_m, s] = match_cores(cores_O4, cb_new, lengthY, lengthX, overlap_thresh, s);  % match_cores

%% Then add these newly found cores to the list of cores and CB's
%cores = imbinarize(cores + cores_m);
%cb = imbinarize(cb + cb_m);   % create updated plot of cb


cores = imbinarize(cores + cores_O4);
cb = imbinarize(cb + cb_new);   % create updated plot of cb


% the rest of the objects are plotted again, to make a new O4_im of
% the "remaining cores"
%we then imopen the remaining objects to see if we can associate them together
%and then run a final "match_remain_cores" function
%which will try to associate the very last O4 objects with the closest cell that is a MAXIMAL distance away (compare to objDAPI)

%if an association is formed, then we add the "core" to the list, and also delete the ObjDAPI from the DAPI list



end

