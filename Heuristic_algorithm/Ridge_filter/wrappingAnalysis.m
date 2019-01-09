function [locFibers, allLengths, s] = wrappingAnalysis(fibers_sub_cb, locFibers, allLengths, siz, minLength, isGreen, dense, s)

%% NEED ALTERNATIVE FOR BEFORE VERSION 2017

% Identifies whether or not a DAPI point is wrapped:
% (1) Cycles through all DAPI, and matches DAPI with CORE
% (2) Then matches CORE with CELL_BODY (cb)
% (3) Then places CELL_BODY one by one back into "fibers_sub_cb" and finds idx of this obj as CELL_IDX
% (4) Then matches CELL_IDX with FIBERS
% (5) Every fiber that is matched is then erased from fibers_idx using cellfun(@) *** NO DOUBLE COUNTING
% (6) Save idx of WRAPPED cell bodies
% (7) The remaining fibers that have not been sorted are then matched to the NEAREST CELL_BODY (cb)
%             ==> (a) by first finding boundary of FIBER with bwboundaries AND the bwboundaries of the CELL_BODIES that ARE WRAPPED
%             ==> (b) then performing pythagoras on every point in "boundaries" to every cell, saving the MIN distance and cell_body_idx
%             ==> (c) then ensure MIN distance < max possible threshold (i.e. fiber can't be ridiculously far away)
%             ==> (d) add fiber count to "Num Sheath"
%
%  Input:
%     objDAPI == linear indices of all DAPI points
%     core ==
%     core_idx ==
%     cb ==
%     cb_idx ==
%
%     fibers_sub_cb == *** image of fibers - cb
%
%     fibers ==
%     fibers_idx ==
%
%     locFibers ==> gone through houghline analysis

%cores_idx =  cores_idx(~cellfun('isempty',cores_idx));   % delete from the list if not a line

multiplierSingleLine = 0.5;  % to be more stringent, that single wrapped fibers must be longer
allWrappedCenters = [];

%% First find and order CB's such that largest are checked FIRST
ordered = cell(0);
idx_ord = 1;

check = zeros(siz);
for Y = 1:length({s.Core})
    if ~isempty(s(Y).Core) && ~isempty(s(Y).CB)  && (s(Y).Bool_W > -1) % if core and CB match and NOT permanently unwrapped
        lengthCB = length(s(Y).CB);
        ordered{idx_ord, 1} = s(Y).Core;
        ordered{idx_ord,2} = Y;         % idx in struct "s"
        ordered{idx_ord, 3} = s(Y).CB;
        ordered{idx_ord,4} = lengthCB;
        idx_ord = idx_ord + 1;
        check(s(Y).CB) = 1;
    end
end

if ~isempty(ordered)  % if there are some cells at least
    %sorted = ordered; % FOR VERSIONS BEFORE 2017
    sorted = sortrows(ordered, 4, 'descend');
    
    
    for i = 1 : length(sorted(:, 1))
        %% (1) Match DAPI + Blobs
        boolFiber = 0;
        idx_s = sorted{i, 2};
        curCore = sorted{i, 1};  % save current pixel idx list
        
        %% RUNNING WITH centerCore = s(idx_s).centerDAPI{1};
        %centerCore = s(idx_s).centerDAPI{1};
        %% OTHERWISE (if running whole_im)
        centerCore = s(idx_s).centerDAPI;
        curCB = sorted{i, 3};
        
        %% (2) Fill lineImage (tmp) with the current DAPI, and then match DAPI/blob with this TMP image
        % THIS PREVENTS OVER-COUNTING in NUM-SHEATHS
        tmp = fibers_sub_cb;
        tmp(curCB) = 1;
%          figure(21); imshow(tmp); title('added Blob');
%          text(centerCore(:, 1) - 20, centerCore(:, 2) + 65, '*',  'color','r' ,'Fontsize',50);
          
        % Make into object after adding CB to "fibers_sub_cb"
        % Then loop through to find this CB + fibers object
        tmp = bwareaopen(tmp, 300);  %%%CLEANS
        CB_fibers = bwconncomp(tmp);
        
        numSheaths = 0;  loc_single_idx = 0;
        whichFibers = cell(1); L = 1;  % to save which fibers are part of this current blob
        for k = 1 : length(CB_fibers.PixelIdxList)
            same = ismember(CB_fibers.PixelIdxList{k}, curCore);  % find the identical value
            
            if ~isempty(find(same, 1))
                curCB_fibers = CB_fibers.PixelIdxList{k};  % saves the object that has the same index
                
                
                %% (3) now have to see if there are any lines in this blobLines (compare to locFibers)
                for j = 1 : length(locFibers)
                    
                    if ~isempty(locFibers{j})
                        % Convert to linear indices
                        findFiber = ismember(curCB_fibers, locFibers{j});  % find the identical value
                        if ~isempty(find(findFiber, 1))
                            boolFiber = 1;
                            numSheaths = numSheaths + 1;
                            whichFibers{L} = j;    % saves the index of the fiber that is associated with the blob
                            
                            loc_single_idx = j;
                            % Saves in struct
                            s(idx_s).Fibers{end + 1} = locFibers{j};
                            
                            % ***THEN SETS THAT FIBER TO BE EMPTY
                            locFibers{j} = [];
                            %allLengths{j} = [];
                            L = L + 1;
                        end
                    end
                end
            end
        end
        
        %% (4) if something only has 1 sheath, then the sheath has to be TWICE AS LONG as usual
        if ~isempty(whichFibers{1}) && dense == 'N'
            singleFiberLength = allLengths{whichFibers{1}};
            if numSheaths == 1 && (singleFiberLength < minLength + minLength * multiplierSingleLine)
                boolFiber = 0;  % or else it is also not counted as wrapped
                if s(idx_s).Bool_W < 1 %%%% IF ON NO PREVIOUS run this was marked as wrapped, then eliminate all the fibers from it too
                    locFibers{loc_single_idx} = s(idx_s).Fibers{1};   % PUTS THE FIBER BACK
                    s(idx_s).Fibers = [];   % Eliminates
               end
            end
        end
        
        %% Saves in struct
        if (boolFiber == 1 || s(idx_s).Bool_W == 1)  % if considered wrapped on a previous run
            s(idx_s).Bool_W = 1;
            allWrappedCenters = [allWrappedCenters; centerCore];
        else  % if MARKED AS PERMANENTLY UNWRAPPED
            s(idx_s).Fibers = [];
        end
    end
    
    figure(5);
    if ~isempty(allWrappedCenters) && ~isGreen
        text(allWrappedCenters(:, 1), allWrappedCenters(:, 2), '*',  'color','g' ,'Fontsize',30);   % writes "wrapped" besides every wrapped neuron
    elseif ~isempty(allWrappedCenters)
        text(allWrappedCenters(:, 1), allWrappedCenters(:, 2), 'wG',  'color','y' ,'Fontsize',3);
    end
    hold on;
end
end