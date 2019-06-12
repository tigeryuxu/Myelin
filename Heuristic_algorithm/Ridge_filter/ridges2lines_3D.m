function [all_lines, locFibers, all_lengths, mask, fibers] = ridges2lines_3D(ridges, siz, hor_factor, minLength, dilate)

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

%% Pre-process:
% (1) Get rid of rectangular objects (MajorAxisLength < MinorAxisLength/3) && MajorAxisLength < 3 * minLength
% ***can NOT be 3 times greater than minLength, b/c may exclude large objs
% that we want to break up for the analysis

cc_ridges = bwconncomp(ridges);
stats_ridges = regionprops3(cc_ridges, 'PrincipalAxisLength', 'VoxelIdxList');

new_ridges = zeros(siz);
for i = 1:length(stats_ridges.VoxelIdxList)
%     if (stats_ridges(i).MinorAxisLength > stats_ridges(i).MajorAxisLength / 2) && (stats_ridges(i).MajorAxisLength < 2 * minLength)
%        continue;
%     else
        new_ridges(stats_ridges.VoxelIdxList{i}) = 1;
%    end
end
fibers = new_ridges;

%% Skeletonize and separate into horizontal + vertical lines
mask = zeros(siz);

%skel = Skeleton3D(logical(fibers));
skel = bwskel(logical(fibers),'MinBranchLength',5);
%figure(); volshow(double(skel));

%skel = bwmorph(new_ridges, 'thin', inf);
branchP = bwmorph3(skel, 'branchpoints');

branchPB = imdilate(branchP, ones(2, 2, 2));   % Expands to 8 connectivity

seg_skel = binarize_3D_otsu(imsubtract(skel, branchPB));


%% 3D comparison of lines: link up any lines that are > 135 degrees from each other
 cc = bwconncomp(seg_skel);
 stats = regionprops3(cc, 'Orientation', 'VoxelList', 'VoxelIdxList');
 
 
 
 %% OR, what if we assume that in 3D space, there is sufficent resolution that
 % one can just eliminate any segments that are small, and leave the longer
 % connected components as lines???
 elim_small_segs = zeros(siz);
 for i = 1:length(stats.VoxelIdxList)
     cur_seg = stats.VoxelIdxList(i);
     if length(cur_seg{1}) > 5
         elim_small_segs(cur_seg{1}) = 1;
     end
 end
 
 reinsert_branch = binarize_3D_otsu(imadd(logical(elim_small_segs), branchPB));

 
 cc = bwconncomp(reinsert_branch);
 stats = regionprops3(cc, 'PrincipalAxisLength', 'VoxelList', 'VoxelIdxList');

 elim_small_segs = zeros(siz);
  for i = 1:length(stats.VoxelIdxList)
     cur_length(1) = stats.PrincipalAxisLength(i);
     cur_seg = stats.VoxelIdxList(i);
     if cur_length(1) > 5
         elim_small_segs(cur_seg{1}) = 1;
     end
 end
 

seg_skel = bwskel(logical(elim_small_segs),'MinBranchLength',5);


% %% Finds orientation of each segment
% cc = bwconncomp(seg_skel);
% stats = regionprops3(cc, 'Orientation', 'VoxelIdxList');
% 
% % Then sort the segments to be similar in some way???
% % Everything above + 45 and below -45 ==> vertical
% 
% [vert, vert_idx] = find([stats.Orientation] > +45   | [stats.Orientation] < -45);
% vert_lines = zeros(siz);
% for i = 1:length(vert_idx)
%     vert_lines(stats.VoxelIdxList{vert_idx(i)}) = 1;
% end
% 
% % Everything else is horizontal
% [hor, hor_idx] = find([stats.Orientation] <= +45   & [stats.Orientation]  >= -45);
% hor_lines = zeros(siz);
% for i = 1:length(hor_idx)
%     hor_lines(stats.VoxelIdxList{hor_idx(i)}) = 1; 
% end
% 
% % Subtract off all the opposite lines to get a fully connected image
% c_vert = binarize_3D_otsu(imsubtract(double(skel), hor_lines));
% c_hort = binarize_3D_otsu(imsubtract(double(skel),vert_lines));
% 
% % Then turn them into list of idx and get length of ellipse to clean up small lines
% %[B,L] = bwboundaries(c_vert, 'noholes');
% cc = bwconncomp(c_vert);
% vv = regionprops3(cc, 'VoxelIdxList', 'PrincipalAxisLength');
% clean_v = zeros(siz);
% N = length(vv.VoxelIdxList);
% while N > 0
%     if vv.PrincipalAxisLength(N, 1) < minLength
%         vv(N, :) = [];
%     else
%         clean_v(vv.VoxelIdxList{N}) = 1;
%     end
%     N = N - 1;
% end
% clean_v = binarize_3D_otsu(clean_v);
% 
% % Option to make lines thicker at the end
% if dilate == 'Y'
%     clean_v = imdilate(clean_v, ones(5, 5));
%     clean_v = imerode(clean_v, ones(10, 1));
% end
% %[B,L] = bwboundaries(clean_v, 'noholes');
% cc = bwconncomp(clean_v);
% vv = regionprops3(cc, 'VoxelIdxList', 'PrincipalAxisLength');
% 
% if hor_factor == 1
%     cc = bwconncomp(c_hort);
%     hh = regionprops3(cc, 'VoxelIdxList', 'PrincipalAxisLength');
%     clean_h = zeros(siz);
%     N = length(hh.VoxelIdxList);
%     while N > 0
%         if hh.PrincipalAxisLength(N) < minLength * hor_factor
%             hh(N, :) = [];
%         else
%             clean_h(hh.VoxelIdxList{N}) = 1;
%         end
%         N = N - 1;
%     end
%     clean_h = binarize_3D_otsu(clean_h);
%     
%     if dilate == 'Y'
%         clean_h = imdilate(clean_h, ones(5, 5));
%         clean_h = imerode(clean_h, ones(1, 10));
%     end
%     cc = bwconncomp(clean_h);
%     hh = regionprops3(cc, 'VoxelIdxList', 'PrincipalAxisLength');
%     
%     all_lines = [vv; hh]; % combines all lines together
% else
%     all_lines = [vv]; % combines all lines together
% end


%% NOW FIND LENGTHS:
% shouild I clean with geodesicfirst??? or just use MajorAxisLength???
all_lines_cc = bwconncomp(seg_skel);  % ADDED LINE IF COMMENT OUT ALL THE TOP
all_lines = regionprops3(all_lines_cc, 'VoxelIdxList', 'PrincipalAxisLength');

c = table2cell(all_lines);
all_lengths = c(:, 2);
locFibers = c(:, 1);

all_lengths = cell(0);
for idx = 1:length(locFibers(:, 1))
    all_lengths{end + 1} = length(locFibers{idx});
end

mask = im2double(seg_skel);

% Give different weights so look different on final mask
%mask(clean_v > 0) = 500;

% if hor_factor == 1
%     mask(clean_h > 0) = 200;
% end

end