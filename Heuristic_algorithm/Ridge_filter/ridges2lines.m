function [all_lines, locFibers, all_lengths, mask, fibers] = ridges2lines(ridges, siz, hor_factor, minLength, dilate)

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
stats_ridges = regionprops(cc_ridges, 'MajorAxisLength', 'MinorAxisLength', 'PixelIdxList');

new_ridges = zeros(siz);
for i = 1:length({stats_ridges.PixelIdxList})
    if (stats_ridges(i).MinorAxisLength > stats_ridges(i).MajorAxisLength / 2) && (stats_ridges(i).MajorAxisLength < 2 * minLength)
       continue;
    else
        new_ridges(stats_ridges(i).PixelIdxList) = 1;
    end
end
fibers = new_ridges;

%% Skeletonize and separate into horizontal + vertical lines
mask = zeros(siz);
skel = bwmorph(new_ridges, 'thin', inf);
branchP = bwmorph(skel, 'branchpoints');
branchPB = imdilate(branchP, ones(3, 3));   % Expands to 8 connectivity

seg_skel = imbinarize(skel - branchPB);

% Finds orientation of each segment
cc = bwconncomp(seg_skel);
stats = regionprops(cc, 'Orientation', 'PixelIdxList');

% Then sort the segments to be similar in some way???
% Everything above + 45 and below -45 ==> vertical

[vert, vert_idx] = find([stats.Orientation] > +45   | [stats.Orientation] < -45);
vert_lines = zeros(siz);
for i = 1:length(vert_idx)
    vert_lines(stats(vert_idx(i)).PixelIdxList) = 1;
end

% Everything else is horizontal
[hor, hor_idx] = find([stats.Orientation] <= +45   & [stats.Orientation]  >= -45);
hor_lines = zeros(siz);
for i = 1:length(hor_idx)
    hor_lines(stats(hor_idx(i)).PixelIdxList) = 1; 
end

% Subtract off all the opposite lines to get a fully connected image
c_vert = imbinarize(skel - hor_lines);
c_hort = imbinarize(skel - vert_lines);

% Then turn them into list of idx and get length of ellipse to clean up small lines
%[B,L] = bwboundaries(c_vert, 'noholes');
vv = regionprops(c_vert, 'PixelIdxList', 'MajorAxisLength');
clean_v = zeros(siz);
N = length(vv);
while N > 0
    if vv(N).MajorAxisLength < minLength
        vv(N) = [];
    else
        clean_v(vv(N).PixelIdxList) = 1;
    end
    N = N - 1;
end
clean_v = imbinarize(clean_v);

% Option to make lines thicker at the end
if dilate == 'Y'
    clean_v = imdilate(clean_v, ones(5, 5));
    clean_v = imerode(clean_v, ones(10, 1));
end
%[B,L] = bwboundaries(clean_v, 'noholes');
vv = regionprops(clean_v, 'PixelIdxList', 'MajorAxisLength');

% hh = regionprops(c_hort, 'PixelIdxList', 'MajorAxisLength');
% clean_h = zeros(siz);
% N = length(hh);
% while N > 0
%     if hh(N).MajorAxisLength < minLength * hor_factor
%         hh(N) = [];
%     else
%         clean_h(hh(N).PixelIdxList) = 1;
%     end
%     N = N - 1;
% end
% clean_h = imbinarize(clean_h);
% clean_h = imdilate(clean_h, ones(5, 5));
% hh = regionprops(clean_h, 'PixelIdxList', 'MajorAxisLength');

%all_lines = [vv; hh]; % combines all lines together
all_lines = [vv]; % combines all lines together

%% NOW FIND LENGTHS:
% shouild I clean with geodesicfirst??? or just use MajorAxisLength???

c = struct2cell(all_lines);
all_lengths = c(1, :);
locFibers = c(2, :);

% Give different weights so look different on final mask
mask(clean_v) = 500;
%mask(clean_h) = 200;

end