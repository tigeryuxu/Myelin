function [cb, no_dilate_cb, s] = cell_body_filt(im, diameter, siz, coreMin, s)

% Cell body filter:
% finds the cell body using bwdist transform
% (1) Uses DIAMETER to threshold out regions of bw that are clustered.
%             i.e. sets to negative everything BELOW diamter threshold
%
% (2) Then THICKENS these areas back to their original size WITHOUT "bridging"
%
% Inputs:
%         im
%
%         diameter
%         objDAPI
% Outputs:
%         cb == cell body image
%
%         cb_idx == cell array of idx of cell bodies


clean_CB = coreMin;


A = im;
A = imcomplement(A);   % invert

objDAPI = {s.objDAPI};

%% Algorithm
D = bwdist(A);  % Euclidean distance
tmp = D;
tmp(tmp > diameter) = 0;    % Take out Euclidean holes

core = D - tmp;  % subtract out the background to get the core
core(core < 0) = 0;

%% Watershed segmentation

% FIRST CLEAN SMALL WEIRD CORES NEXT TO LARGE ONES THOUGH???
% dilate them together maybe??? or... no idea...

D = -core;  % EDT
mask = imextendedmin(D, 10);   % Extended minima
D2 = imimposemin(D, mask);

Ld2 = watershed(D2);
bw3 = core;
bw3(Ld2 == 0) = 0;
bw = bw3;

%% Display the labelled watershed and draw boundaries
figure(102);
[B,L] = bwboundaries(bw, 'noholes');
imshow(bw);
%imshow(label2rgb(L, @jet, [.5 .5 .5]));
title('Watershed CBs');
hold on

no_dilate_cb = bw;


%% CLEAN UP EVERYTHING THAT IS SMALL (i.e. DAPI on tiny CBs)

% but SHOULD KEEP, if the cell is associated with O4+ objects > than
% certain size, and if it is the ONLY CB within that large O4+ object...


%% WHAT WE WANT IS TO keep the CB, but NOT do "match_cores" for CBs less than Clean_CB

% ***need to loop, b/c want to get rid of associated "Core" as well
objList = bwconncomp(bw);
cb_idx = objList.PixelIdxList;

for T = 1:length(cb_idx)
    curCB = cb_idx{T};
    
    if length(curCB) < clean_CB   % TOO SMALL
        bw(curCB) = 0;  % deletes the CB
        
        for Y = 1:length(s)
            curCore = s(Y).Core;
            same = ismember(curCB, curCore);
            if ~isempty(find(same, 1))
                s(Y).Core = [];   % also deletes the core
            end
        end
    end
end


%% THICKEN
cb = bwmorph(bw, 'thicken', diameter);  % Then thicken the edges back to the original ledge

%% Save:
obj = bwconncomp(cb);
cb_idx = obj.PixelIdxList;

%% Associate CB with objDAPI:
for i = 1:length(objDAPI)
    curDAPI = objDAPI{i};
    for j = 1:length(cb_idx)
        cur_cb = cb_idx{j};
        same = ismember(cur_cb, curDAPI);
        if ~isempty(find(same, 1))
            [s(i).CB] = [s(i).CB; cur_cb];     %% ADDS ALL CB
        end
    end
end

end