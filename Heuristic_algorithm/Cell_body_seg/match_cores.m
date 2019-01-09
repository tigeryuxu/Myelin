function [new_cores, new_cb, s] =match_cores(cores, cb, siz, s)

% If a cell body does not have a "core" associated with it (belonging to a DAPI)
% then, we check to see if there is ANY DAPI touching the cell body
% b/c it is likely that the cell is the one that is wrapped (as there is nothing else in the vicinity)
%
%     also, by the end, the rest of the cb are "bridged" if they don't belong to any DAPI
%     b/c if they're next to another cb, then they should probably be connected instead
%
% Inputs:
%
%
%
% Outputs:
%     new_cores ==> updates the image
%     new_cores_idx ==> as well as the list of cores

cores_idx = {s.Core};
cb_idx = {s.CB};

D = bwdist(~cb);

new_cb = zeros(siz); % Blank image
new_cores = cores;

objDAPI = {s.objDAPI};

for i = 1:length(cb_idx)
    
    %% Check each "cb" to see if associates with a "core"
    cur_cb = cb_idx{i};
    match = 0;
    for j = 1:length(cores_idx)
        cur_core = cores_idx{j};
        same = ismember(cur_cb, cur_core);
        if ~isempty(find(same, 1)) 
            match = 1;
        end
    end
    
    %% If NO matching "core", check to see if there is matching DAPI
    if match == 0
        all_DAPI_idx = cell(1); idx = 1;
        for k = 1:length(objDAPI)
            curDAPI = objDAPI{k};
            same = ismember(cur_cb, curDAPI);
            if ~isempty(find(same, 1))
                all_DAPI_idx{1, idx} = k;
                all_DAPI_idx{2, idx} = max(D(curDAPI));   % maximal value of EDT for given curDAPI idx
                idx = idx + 1;
            end
        end
        
        % If there is matching DAPI, then add the "cb" to the image
        if ~isempty(all_DAPI_idx{1})
            % Finds DAPI with highest EDT idx values (closest to center)
            [val, T] = max([all_DAPI_idx{1, :}]);
            
            %% Adds THE OVERLAPING PART as a new core in "cores_idx" list
            
            
            DAPI = objDAPI{all_DAPI_idx{1, T}};
            same = ismember(DAPI, cur_cb);
            overlap_idx = find(same);
            overlap = DAPI(overlap_idx);
            
            s(all_DAPI_idx{1, T}).Core = overlap;
            % adds to image
            new_cb(cur_cb) = 1;
            new_cores(overlap) = 1;
            
        end
        
        %% Else, if it DOES match with pre-exisiting "Core", just add the cb to the new_cb
    else
        new_cb(cur_cb) = 1;
    end
    
end

end