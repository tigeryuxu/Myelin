function [s_overall, together, only_branched_fibers, final_no_sub_fibers] = line_traversal_3D(ridges, siz, minLength, dilate)

%Skeletonize fibers, and find junctions
%then trace lines going in certain orientation, maybe split into segments via junctions?
%and then use orientation to find ones nearby (like 2 pixels away) with VERY SIMILAR directional orientation and re-connect them
%then delete all of the segments that are TOO SMALL
%
% Input:
%         ridges = bw image of co-localized(???) and cleaned from houghlines (???) ridges
%
% Output:
%         lines = bw image of segmented LINES
%         mask = skeleton mask of lines
%
%         prints out overlay image of mask over lines


%% THINGS TO DO:
% (1) Must get average vector of ENTIRE fiber, not just the end tip
% (2) Figure out next step

%% Pre-process:

thresh_angle = 120



bw_ridges = imbinarize(ridges);
%skel = Skeleton3D(logical(fibers));
skel = bwskel(logical(bw_ridges),'MinBranchLength',5);
%% NOT NECESSARY TO PRUNE B/C these segmentations are quite smooth

%skel = bwmorph(new_ridges, 'thin', inf);
branchP = bwmorph3(skel, 'branchpoints');
unsub_skel = skel;

%% step (0): subtract out ALL branch points from original skeleton
skel = imsubtract(skel, branchP);
cc_skel = bwconncomp(skel);

%% Create crops around branch points and collect these in a large struct at the end
cc_branchP = bwconncomp(branchP);
branchP_idx = regionprops3(cc_branchP, 'Orientation', 'VoxelIdxList', 'VoxelList', 'Centroid');

im_size = siz;
all_branch_point_crops = [];

s_overall = struct('sheath_idx', []);
for idx = 1:length(branchP_idx.Centroid(:, 1))
    
       bp_center = branchP_idx.Centroid(idx, :);
       
       % only want to consider 1 branch point at a time
       bp_vox_list = branchP_idx.VoxelIdxList{idx, :};
       bp_tmp = zeros(size(branchP));
       bp_tmp(bp_vox_list) = 1;
    
       bp_x = bp_center(1);
       bp_y = bp_center(2);
       bp_z = bp_center(3);
       
       
       box_x_max = bp_x + 30; box_x_min = bp_x - 30;
       box_y_max = bp_y + 30; box_y_min = bp_y - 30;
       box_z_max = bp_z + 10; box_z_min = bp_z - 10;
             
       im_size_x = im_size(1);
       im_size_y = im_size(2);
       im_size_z = im_size(3);
       
       if box_x_max > im_size_x
            overshoot = box_x_max - im_size_x
            box_x_max = box_x_max - overshoot;
            box_x_min = box_x_min - overshoot;
       end
       
       if box_x_min <= 0
           overshoot_neg = (-1) * box_x_min + 1
           box_x_min = box_x_min + overshoot_neg;
           box_x_max = box_x_max + overshoot_neg;
       end
       
 
        if box_y_max > im_size_y
            overshoot = box_y_max - im_size_y
            box_y_max = box_y_max - overshoot;
            box_y_min = box_y_min - overshoot;
       end
       
       if box_y_min <= 0
           overshoot_neg = (-1) * box_y_min + 1
           box_y_min = box_y_min + overshoot_neg;
           box_y_max = box_y_max + overshoot_neg;
       end
       
       
       
       if box_z_max > im_size_z
            overshoot = box_z_max - im_size_z
            box_z_max = box_z_max - overshoot;
            box_z_min = box_z_min - overshoot;
       end
       
       if box_z_min <= 0
           overshoot_neg = (-1) * box_z_min + 1
           box_z_min = box_z_min + overshoot_neg;
           box_z_max = box_z_max + overshoot_neg;
       end
       
       box_x_max - box_x_min
       box_y_max - box_y_min
       box_z_max - box_z_min
       
       
       % CROP IT
       cropped_skel = skel(box_y_min:box_y_max, box_x_min:box_x_max, box_z_min:box_z_max);
       cropped_bp = bp_tmp(box_y_min:box_y_max, box_x_min:box_x_max, box_z_min:box_z_max);
       
       %% TIGER - crop unsub_skel for figures for paper
       %crop_unsub_skel = unsub_skel(box_y_min:box_y_max, box_x_min:box_x_max, box_z_min:box_z_max);
       %figure(35); volshow(double(crop_unsub_skel));

       %figure(37); volshow(im2double(cropped_skel))
       %figure(38); volshow(im2double(cropped_bp))
       
       
       %% (2) DILATE THE BP to and eliminate all things NOT touching this dilated bp
       cropped_bp_small = imdilate(cropped_bp, strel('sphere', 2));   % Expands to 8 connectivity
       
       cc_skel = bwconncomp(cropped_skel);
       skel_idx = regionprops3(cc_skel, 'VoxelIdxList');

       cc_crop_bp = bwconncomp(cropped_bp_small);
       crop_bp_idx = regionprops3(cc_crop_bp, 'VoxelIdxList');
       
       cleaned_skel = zeros(size(cropped_skel));
       num_sheaths = 0;
       for T = 1:length(skel_idx.VoxelIdxList)
           if iscell(skel_idx.VoxelIdxList)
               skel_idx.VoxelIdxList;
               cur_fiber = skel_idx.VoxelIdxList{T};
               match = 0;
               for Y = 1:length(crop_bp_idx.VoxelIdxList)
                   cur_bp = crop_bp_idx.VoxelIdxList{Y};
                   same = ismember(cur_fiber, cur_bp);
                   if ~isempty(find(same, 1))
                       cleaned_skel(cur_fiber) = 1;
                       num_sheaths = num_sheaths + 1;
                   end
               end
           end
       end

      
       %% (3) use same small dilated bp to subtract it out from the middle (detaches any connections)
       skel_sub_bp = imsubtract(imbinarize(cleaned_skel), imbinarize(cropped_bp_small));
       skel_sub_bp = imbinarize(skel_sub_bp);
      
       % skip this BP if only have 1 branching skeleton here
       cc_skel = bwconncomp(skel_sub_bp);
       skel_sub_idx = regionprops3(cc_skel, 'VoxelIdxList');
       if length(skel_sub_idx.VoxelIdxList) <= 2
           length(skel_sub_idx.VoxelIdxList)
           
           % but also do add these fibers to the overall count
           labelled_connected_lines = zeros(size(skel_sub_bp));
           % combine indicies to make a labelled colorful array
           for like_idx = 1:length(skel_sub_idx.VoxelIdxList)
               if iscell(skel_sub_idx.VoxelIdxList)
                   combine_indices = skel_sub_idx.VoxelIdxList{like_idx};
                   labelled_connected_lines(combine_indices) = 1;
               end
           end
           all_sheath_values = unique(labelled_connected_lines)
           
           % Then insert this colorful cropped array back into an array of size of input image
          blank_full_size = zeros(size(skel));
          blank_full_size(box_x_min:box_x_max, box_y_min:box_y_max, box_z_min:box_z_max) = labelled_connected_lines;
          
          %figure(88); volshow(labelled_connected_lines);
          
          %  Now get the indicies of the sheaths in the NON-cropped image using colorful labelled indexing
          % and save in OVERALL struct (skip val_index == 0)
          for val_idx = 2:length(all_sheath_values)
              cur_fiber_idx = find(blank_full_size == all_sheath_values(val_idx));
              s_overall(end + 1).sheath_idx = cur_fiber_idx;
          end
          
          continue; 
       end
       
       
       %% (4) DILATE BP even further but set everything OUTSIDE that range to ZERO
       skel_vectors = cleaned_skel;
       % so now get short line segments leading INTO the bp
       cropped_bp_large = imdilate(cropped_bp, strel('sphere', 6));
       skel_vectors(cropped_bp_large == 0) = 0;
       
       %% (5) now get START (closest to bp) and END points by using 2 spheres
       cropped_bp_start = imdilate(cropped_bp, strel('sphere', 2));
       cropped_bp_start(skel_vectors == 0) = 0;
       skel_vectors(cropped_bp_start == 1) = 2;
       skel_start = skel_vectors;
       skel_start(skel_vectors == 1) = 0;
       
       cropped_bp_end = imdilate(cropped_bp, strel('sphere', 5));
       %cropped_bp_end(skel_vectors == 0) = 0;
       skel_ends = skel_vectors;
       skel_ends(cropped_bp_end == 1) = 0;
       
       skel_vectors(skel_ends == 1) = 3;
       figure(36); volshow(im2double(skel_vectors));
       
       %% (6) go through each fiber and save the index of the entire sheaths AND start/end
       c= cell(length(skel_sub_idx.VoxelIdxList), 1); % initializes Bool_W with all zeros
       %[c{:}] = deal(0);
       s = struct('sheath_idx', c, 'start_idx', c, 'end_idx', c, 'vector', c, 'closest_match', c);
       
       for t = 1:length(skel_sub_idx.VoxelIdxList)
          cur_fiber = skel_sub_idx.VoxelIdxList{t};
          
          mask_tmp = zeros(size(skel_vectors));
          mask_tmp(cur_fiber) = 1;
          
          tmp_skel = skel_vectors;
          tmp_skel(mask_tmp == 0) = 0;
          
          % get start/end
          start_idx = find(tmp_skel == 2, 1,  'first')  % START
          if isempty(start_idx) 
              start_idx = find(tmp_skel == 1, 1, 'first')
          end
          [s_x, s_y, s_z] = ind2sub(im_size, start_idx);
          
          end_idx = find(tmp_skel == 3, 1, 'last')  % end index
          if isempty(end_idx) 
              empty_idx = find(tmp_skel == 1, 1, 'last') 
          end
          [e_x, e_y, e_z] = ind2sub(im_size, end_idx);

          s(t).sheath_idx = cur_fiber;
          s(t).start_idx = [s_x, s_y, s_z];
          s(t).end_idx = [e_x, e_y, e_z];
          s(t).vector = s(t).end_idx - s(t).start_idx
           
       end
       
       
       %% (7) Calculate theta between each sheath and corresponding sheaths
       % theta = arcos[(u * v)/|u||v|] 
       skip_indices = [];
       for idx_sheath = 1:length(skel_sub_idx.VoxelIdxList)
           skip_indices = [skip_indices, idx_sheath];
           all_thetas = [];
           for other_sheath_idx = 1:length(skel_sub_idx.VoxelIdxList)
               if other_sheath_idx == skip_indices  % skip so don't compare with self
                   all_thetas = [all_thetas, 0];
                   continue;
               end
               
                              
               % find dot product u * v
               u = s(idx_sheath).vector;
               v = s(other_sheath_idx).vector;
               
               if isempty(u) || isempty(v)
                   continue;
               end
               
               dot_prod = dot(u, v);
               
               % find magnitude of each vector then multiply |u||v|
               mag_u = norm(u);
               mag_v = norm(v);
               
               % find arccos
               theta = acos(dot_prod/(mag_u * mag_v));
               degrees = round(theta * (180/pi))
               all_thetas = [all_thetas, degrees];
           end
           
           [val, idx] = max(all_thetas)
           
           %% FIND the index of the other sheath that has theta > thresh angle (120 deg)
           %% IF there is NOT other sheath 120 deg away, then will set index value == 0
           if val >= thresh_angle
               s(idx_sheath).closest_match = idx;
               skip_indices = [skip_indices, idx];  % SKIP THIS INDEX IN THE FUTURE becuase already found one that matches!!!
           else
               s(idx_sheath).closest_match = 0;  
           end
       end

       labelled_connected_lines = zeros(size(skel_vectors));
       %% (8) combine "like" indicies to make a labelled colorful array
       for like_idx = 1:length(skel_sub_idx.VoxelIdxList)
           combine_indices = s(like_idx).sheath_idx;
           idx_close_match = s(like_idx).closest_match;
           if ~isempty(idx_close_match) && idx_close_match ~= 0
                combine_indices = [combine_indices; s(idx_close_match).sheath_idx];
           elseif idx_close_match == 0
                combine_indices = combine_indices;  % add nothing if it did not match with anything else
           end
           labelled_connected_lines(combine_indices) = like_idx;
       end
       all_sheath_values = unique(labelled_connected_lines)
       
       %% (9) Then insert this colorful cropped array back into an array of size of input image
       blank_full_size = zeros(size(skel));
       blank_full_size(box_x_min:box_x_max, box_y_min:box_y_max, box_z_min:box_z_max) = labelled_connected_lines;
       
       figure(88); volshow(labelled_connected_lines);
       
       %% (10) Now get the indicies of the sheaths in the NON-cropped image using colorful labelled indexing
       %% and save in OVERALL struct (skip val_index == 0)
       for val_idx = 2:length(all_sheath_values)
           cur_fiber_idx = find(blank_full_size == all_sheath_values(val_idx));
           s_overall(end + 1).sheath_idx = cur_fiber_idx;
       end
       
       
end

%% (11) At the end, use OVERALL struct to combine fiber segments that have identical indicies!!!
full_size_restored_disconnected = zeros(size(skel));
new_combined_s_overall = struct('sheath_idx', []);
for i = 1:length(s_overall)
   cur_fiber = s_overall(i).sheath_idx;
   
   for Y = 1:length(s_overall)
      if Y == i
          continue
      end
      next_fiber = s_overall(Y).sheath_idx; 
      
      % IF THEY MATCH, COMBINE THEM TOGETHER
      if ismember(cur_fiber, next_fiber)
         cur_fiber = [cur_fiber; next_fiber]; 
         s_overall(Y).sheath_idx = [];  %% ERASES THIS MATCHING FIBER FROM THE OVERALL LIST
         Y
      end
   end
   new_combined_s_overall(end + 1).sheath_idx = cur_fiber;
   full_size_restored_disconnected(cur_fiber) = i;
end
%figure(99); volshow(im2double(full_size_restored_disconnected), 'BackgroundColor', [0,0,0]);title('full');

together = imadd(skel, full_size_restored_disconnected);
together_show = imadd(skel, full_size_restored_disconnected);
%figure(101); volshow(im2double(together_show), 'BackgroundColor', [0,0,0]);title('full');

only_branched_fibers = full_size_restored_disconnected;

%% Also get a mask of all the fibers that did NOT have branch points
cc_skel = bwconncomp(unsub_skel);
skel_idx = regionprops3(cc_skel, 'VoxelIdxList');
final_no_sub_fibers = zeros(size(skel));
for T = 1:length(skel_idx.VoxelIdxList)
    if iscell(skel_idx.VoxelIdxList)
        skel_idx.VoxelIdxList;
        cur_fiber = skel_idx.VoxelIdxList{T};
        match = 0;
        for Y = 1:length(s_overall)
            cur_bp = s_overall(Y).sheath_idx;
            same = ismember(cur_fiber, cur_bp);
            if isempty(find(same, 1))
                final_no_sub_fibers(cur_fiber) = 1;
            end
        end
    end
end
%figure(102); volshow(im2double(final_no_sub_fibers), 'BackgroundColor', [0,0,0]);title('no sub');


end




