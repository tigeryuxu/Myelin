function [all_internodes, one_node, two_nodes] = find_internodes(greenImage, mask, DAPIsize, DAPImetric, enhance_DAPI, internode_size, im_size, hor_factor, minLength, dil_lines, cur_dir, saveDirName, filename_raw, fileNum_sav)

%% COPY code after this point into other script
NaVimage = greenImage;
coloc_im = imfuse(NaVimage, mask);
figure(8); imshow(coloc_im);


%% Identify internodes
internode_size = 5;
[mat, internd, bw_internd]= DAPIcount_2(NaVimage, DAPIsize, DAPImetric, enhance_DAPI, internode_size);

diff = imfuse(bw_internd, NaVimage);

%% fuse internodes and fibers image
mask = imdilate(mask, strel('disk', 2));
coloc_bw = imfuse(bw_internd, mask);
figure(9); imshow(coloc_bw); title('Raw Nodes and Fibers');

%%  %%%%%%PROBLEM:
%to count lengths, need to then match this mask with the original "locFibers" and separate those into individual fibers...
%the problem is that horizontal vs. vertical are different.
%maybe could just run ridges2lines again???

%% (1) Find and subtract out the internodes, and then erode
% BUT, want to maintain different horizontal vs. vertical lines, so must manually set to zero
obj_intnd = bwconncomp(bw_internd);
intnd_idx = obj_intnd.PixelIdxList;
sub_mask = mask;
for i = 1:length(intnd_idx)
    sub_mask(intnd_idx{i}) = 0;
end
figure(10); imshow(sub_mask, []); title('Subtracted nodes');

% Then find endpoints and dilate them out
skel = bwmorph(sub_mask, 'skel', inf);
end_ps = bwmorph(skel, 'endpoints');

big_end_ps = imdilate(end_ps, strel('disk', 2));

% Then match these dilated endpoints with the NaV1 stain
[cBin, rBin] = find(big_end_ps);
binInd = [rBin, cBin];
[rRed, cRed] = find(bw_internd);
redInd = [cRed, rRed];
inter = intersect(binInd, redInd, 'rows');
tmpLines = zeros(im_size);   % temporary array
for test = 1: length(inter)
    tmpLines(inter(test, 2), inter(test, 1)) = 1;   % plots all the intersecting red lines
end

new_internds = bw_internd;
for Y = 1:length(intnd_idx)
    if tmpLines(intnd_idx{Y}) == 0
        new_internds(intnd_idx{Y}) = 0;
    end
end
coloc_sub = imfuse(new_internds, sub_mask);
figure(11); imshow(coloc_sub);
all_internodes = coloc_sub;

[a_all_lines, a_locFibers, a_tmpLength, a_mask, a_fibers]  = ridges2lines(sub_mask, im_size, hor_factor, minLength, dil_lines);


%% (2) Only keep fibers that coloc with 1 internode
one_node_mask = a_mask;

% Dilate the internodes to improve overlap
new_internds_dil = imdilate(new_internds, strel('disk', 3));
new_obj = bwconncomp(new_internds_dil);
new_intnd_idx = new_obj.PixelIdxList;

new_obj = bwconncomp(one_node_mask);
new_sub_idx = new_obj.PixelIdxList;

new_sub_idx = a_locFibers;  % use locFibers

for T = 1:length(new_sub_idx)
    cur_fiber = new_sub_idx{T};
    match = 0;
    for Y = 1:length(new_intnd_idx)
        cur_intnd = new_intnd_idx{Y};
        same = ismember(cur_fiber, cur_intnd);
        if ~isempty(find(same, 1))
            match = match + 1;
        end
    end
    if match < 1
        tmp = zeros(im_size);   % Dilate b/c locFibers is a skeleton
        tmp(cur_fiber) = 1;
        tmp = imdilate(tmp, strel('disk', 4));
        one_node_mask(find(tmp)) = 0;
    end
end
one_node = one_node_mask;
one_node_mask = imdilate(one_node_mask, strel('disk', 2));
coloc_sub_2 = imfuse(new_internds, one_node_mask);
figure(12); imshow(coloc_sub_2); title('> 1 internode');


%% (3) Only keep the ones that match with 2 internodes
new_internds_dil = imdilate(new_internds, strel('disk', 3));
new_obj = bwconncomp(new_internds_dil);
new_intnd_idx = new_obj.PixelIdxList;

two_node_mask = a_mask;
new_obj = bwconncomp(two_node_mask);
new_sub_idx = new_obj.PixelIdxList;


new_sub_idx = a_locFibers;  % use locFibers
for T = 1:length(new_sub_idx)
    cur_fiber = new_sub_idx{T};
    match = 0;
    for Y = 1:length(new_intnd_idx)
        cur_intnd = new_intnd_idx{Y};
        same = ismember(cur_fiber, cur_intnd);
        if ~isempty(find(same, 1))
            match = match + 1;
        end
    end
    if match < 2
        tmp = zeros(im_size);   % Dilate b/c locFibers is a skeleton
        tmp(cur_fiber) = 1;
        tmp = imdilate(tmp, strel('disk', 4));
        two_node_mask(find(tmp)) = 0;
    end
end
two_nodes = two_node_mask;
two_node_mask = imdilate(two_node_mask, strel('disk', 2));
coloc_sub_3 = imfuse(new_internds, two_node_mask);
figure(13); imshow(coloc_sub_3); title('> 2 internodes');


%% Save results
name = erase(filename_raw, '.tif');

cd(saveDirName);
figure(9);
filename = strcat('Result', num2str(fileNum_sav), '_', filename_raw, '_', '_Raw nodes and fibers');
print(filename,'-dpng'); hold off;

figure(12);
filename = strcat('Result', num2str(fileNum_sav), '_', filename_raw, '_', '_Single node') ;
print(filename,'-dpng'); hold off;

figure(13);
filename = strcat('Result', num2str(fileNum_sav), '_', filename_raw, '_', '_Double nodes') ;
print(filename,'-dpng'); hold off;
cd(cur_dir);

end