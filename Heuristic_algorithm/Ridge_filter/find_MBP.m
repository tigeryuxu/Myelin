function [cores_MBP, s] = find_MBP(s, O4_adapt, switch_sheaths, O4_original, MBP_im, fillHoles, enhance, greenOrig, get_ensheathed_only, subtract_old_MBP)

    % First find cores of ENSHEATHED cells to use for "imposemin"
    if get_ensheathed_only == 1
        tmp_CBs = zeros(size(O4_adapt));
        for cell_num = 1:length(s(:, 1))
            if s(cell_num).Bool_W
                CB_area = s(cell_num).CB;
                tmp_CBs(CB_area) = 1;
            end
        end
    else  % OTHERWISE, get the MBP area for all other non-ensheathed cells!
        tmp_CBs = zeros(size(O4_adapt));
        for cell_num = 1:length(s(:, 1))
            if s(cell_num).Bool_W == 0 && s(cell_num).O4_bool == 1
                CB_area = s(cell_num).CB;
                tmp_CBs(CB_area) = 1;
            end
        end
    end
    
    %O4_adapt = adapthisteq(O4_original);
    if switch_sheaths == 0
        O4_adapt = O4_original;
    end
    [combined_im, originalRed] = imageAdjust(O4_adapt, fillHoles, enhance);

    bw = ~bwareaopen(~combined_im, 10);  % clean
    D = -bwdist(~bw);  % EDT
    D2 = imimposemin(D, tmp_CBs);

    Ld2 = watershed(D2);
    bw3 = bw;
    bw3(Ld2 == 0) = 0;
    bw = bw3;

    figure(120); title('CB watershed');
    [B,L] = bwboundaries(bw, 'noholes');
    imshow(bw);
    imshow(label2rgb(L, @jet, [.5 .5 .5]));
    hold on;

    % Colocalize with CBs and only keep the remainder
    obj_CBs = bwconncomp(bw);
    cb_CB_idx  = obj_CBs.PixelIdxList;

    cores_CB = zeros(size(bw));
    idx_new = cell(0);
    for Y = 1:length(cb_CB_idx)
        cur_CB = cb_CB_idx{Y};
        if isempty(cur_CB)   %% SPEED UP CODE BY REDUCING REDUNDANCY
            continue;
        end
        for T = 1:length({s.CB})
            % if get_ensheathed_only, then only get the MBP of the
            % ENSHEATHED cells, otherwise, get the O4+ cells only
            if get_ensheathed_only == 1  
                bool = (s(T).Bool_W == 1);            
            else
                bool = (s(T).Bool_W == 0 && s(T).O4_bool == 1); 
            end
            
            % begin checking the bool
            if bool
                CB_obj = s(T).CB;
                same = ismember(CB_obj, cur_CB);
                if ~isempty(find(same, 1))
                    overlap_idx = find(same);
                    overlap = CB_obj(overlap_idx);
                    cores_CB(cur_CB) = T;
                    break;
                end
            end
        end
    end
    if get_ensheathed_only
        figure(121);
    else
        figure(124);
    end
    title('CB watershed ensheathed');
    %[B,L] = bwboundaries(cores_CB, 'noholes');
    %imshow(cores_CB);
    imshow(label2rgb(cores_CB, @lines, [.5 .5 .5]));
    hold on;

    figure(122); title('Ensheathed Cores');
    imshow(tmp_CBs);


    %% CORRELATE NOW WITH MBP
    bw_MBP = imbinarize(greenOrig);
    bw_MBP(~cores_CB) = 0;

    obj_MBP = bwconncomp(bw_MBP);
    cb_MBP_idx  = obj_MBP.PixelIdxList;

    obj_CB_E = bwconncomp(cores_CB);
    cb_CB_E_idx = obj_CB_E.PixelIdxList;

    cores_MBP = zeros(size(bw));
    idx_new = cell(0);
    for Y = 1:length(cb_MBP_idx)
        cur_MBP = cb_MBP_idx{Y};
        if isempty(cur_MBP)   %% SPEED UP CODE BY REDUCING REDUNDANCY
            continue;
        end
        for T = 1:length(cb_CB_E_idx)
            MBP_obj = cb_CB_E_idx{T};
            same = ismember(MBP_obj, cur_MBP);
            if ~isempty(find(same, 1))
                overlap_idx = find(same);
                overlap = MBP_obj(overlap_idx);
                cores_MBP(cur_MBP) = T;
                break;
            end

        end
    end
    
    if get_ensheathed_only
        figure(123);
    else
        figure(125);
    end

    %% IF get only the NON-ensheathed cores, subtract out the original MBP image
    if get_ensheathed_only == 0
        cores_MBP(imbinarize(subtract_old_MBP)) = 0;
    end
    
    title('MBP of ensheathed');
    imshow(label2rgb(cores_MBP, @lines, [.5 .5 .5]));
    hold on;


    
    %% NOW LOOP THROUGH AND COUNT HOW MUCH MBP PER CELL. THEN SAVE the results!
    cell_numbers = unique(cores_MBP);
    area_per_cell = [];
    for T = 2:length(cell_numbers)
        a = length(cores_MBP(cores_MBP == T - 1));   % gets the area
        area_per_cell = [area_per_cell, a];    % ***AREA might be ZERO ==> b/c not ADAPTHISTEQ
    end

    if get_ensheathed_only == 1
        s(1).AreaOverall = area_per_cell;
    else
        s(2).AreaOverall = area_per_cell;
    end
end