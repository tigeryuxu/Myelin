function [all_internodes, all_caspr_coloc, one_node, one_node_caspr, two_nodes, two_nodes_caspr, bw_CASPR] = find_internodes_3D(greenImage, mask, DAPIsize, DAPImetric, enhance_DAPI, internode_size, im_size, hor_factor, minLength, dil_lines, cur_dir, saveDirName, filename_raw, fileNum_sav)

dil_amount_CASPR = 6
dil_amount_myelin = 1
dil_amount_endpoint = 6

%% COPY code after this point into other script
NaVimage = greenImage;
%coloc_im = imfuse(NaVimage, mask);
%figure(8); imshow(coloc_im);


%% Identify internodes
%[mat, internd, bw_internd]= DAPIcount_3D(NaVimage, DAPIsize, DAPImetric, enhance_DAPI, internode_size);

bw_CASPR = NaVimage;

if bw_CASPR == 1   % IF ONLY HAVE A SINGLE VALUE (means blank) ==> set to blank
    bw_CASPR = zeros(size(bw_CASPR));
end
%diff = imfuse(bw_internd, NaVimage);

%% fuse internodes and fibers image
%mask = imdilate(mask, strel('disk', 2));
%coloc_bw = imfuse(bw_internd, mask);
%figure(9); imshow(coloc_bw); title('Raw Nodes and Fibers');

%%  %%%%%%PROBLEM:
%to count lengths, need to then match this mask with the original "locFibers" and separate those into individual fibers...
%the problem is that horizontal vs. vertical are different.
%maybe could just run ridges2lines again???

%% (1) Find and subtract out the internodes, and then erode
% BUT, want to maintain different horizontal vs. vertical lines, so must manually set to zero
%obj_CASPR = bwconncomp(bw_CASPR);
%CASPR_idx = obj_CASPR.PixelIdxList;
mask_MBP = mask;
%for i = 1:length(intnd_idx)
%    sub_mask(intnd_idx{i}) = 0;
%end
figure(10); volshow(im2double(mask_MBP), 'BackgroundColor', [0,0,0]);title('Subtracted nodes');

% Then find endpoints and dilate them out
mask_MBP = logical(mask_MBP);
skel = bwskel(mask_MBP);
end_ps = bwmorph3(skel, 'endpoints');

big_end_ps = imdilate(end_ps, strel('sphere', dil_amount_endpoint));

% Then match these dilated endpoints with the NaV1 stain'
end_point_idx = find(big_end_ps);
CASPR_idx = find(bw_CASPR);
inter = intersect(end_point_idx, CASPR_idx, 'rows');
tmpLines = zeros(im_size);   % temporary array
tmpLines(inter) = 1;   % temporary array

% [cBin, rBin] = find(big_end_ps);
% binInd = [rBin, cBin];
% [rRed, cRed] = find(bw_internd);
% redInd = [cRed, rRed];
% inter = intersect(binInd, redInd, 'rows');
% tmpLines = zeros(im_size);   % temporary array
% for test = 1: length(inter)
%     tmpLines(inter(test, 2), inter(test, 1)) = 1;   % plots all the intersecting red lines
% end

new_CASPR = tmpLines;
% for Y = 1:length(CASPR_idx)
%     if tmpLines(CASPR_idx{Y}) == 0
%         new_CASPR(CASPR_idx{Y}) = 0;
%     end
% end
%coloc_sub = imfuse(new_internds, sub_mask);
%figure(11); imshow(coloc_sub);
all_internodes = mask_MBP;
all_caspr_coloc = new_CASPR;
%sub_mask = binarize_3D_otsu(sub_mask);
%[a_all_lines, a_locFibers, a_tmpLength, a_mask, a_fibers]  = ridges2lines_3D(mask_MBP, im_size, hor_factor, minLength, dil_lines);
%figure(); volshow(a_mask, 'BackgroundColor', [0,0,0]);title('Subtracted nodes');

a_mask = mask_MBP;


%% Eliminate round/short sheaths
cc = bwconncomp(all_internodes);
vv = regionprops3(cc, 'PrincipalAxisLength', 'VoxelIdxList');

elim_short_intern = zeros(size(all_internodes));
for T = 1:length(vv.VoxelIdxList)
    
    if length(vv.VoxelIdxList{T}) > 100 && vv.PrincipalAxisLength(T,1) > 20
        elim_short_intern(vv.VoxelIdxList{T}) = 1;
    end
end
all_internodes = elim_short_intern;





%% (2) Only keep fibers that coloc with 1 internode
one_node_MBP = a_mask;

% Dilate the internodes to improve overlap
%new_internds_dil = imdilate(new_internds, strel('sphere', dil_amount_myelin));
new_CASPR_dil = new_CASPR;

%% TRY USING ENDPOINTS to eliminate new_internodes? - 28/04/2019
bw = one_node_MBP;
%bw(bw > 0) = 1;
skel = bwskel(bw);
endpoints = bwmorph3(skel, 'endpoints');
endpoints = imdilate(endpoints, strel('sphere', dil_amount_endpoint));
new_CASPR_dil(endpoints < 1) = 0; 


new_CASPR_dil = imdilate(new_CASPR_dil, strel('sphere', dil_amount_CASPR));
new_obj = bwconncomp(new_CASPR_dil);
new_CASPR_idx = new_obj.PixelIdxList;

new_obj = bwconncomp(one_node_MBP);
new_MBP_idx = new_obj.PixelIdxList;

%new_MBP_idx = a_locFibers;  % use locFibers

one_node_MBP_NEW = zeros(im_size);
for T = 1:length(new_MBP_idx)
    cur_fiber = new_MBP_idx{T};
    match = 0;
    for Y = 1:length(new_CASPR_idx)
        cur_intnd = new_CASPR_idx{Y};
        same = ismember(cur_fiber, cur_intnd);
        if ~isempty(find(same, 1))
            match = match + 1;
        end
    end
    %     if match < 1
    %         tmp = zeros(im_size);   % Dilate b/c locFibers is a skeleton
    %         tmp(cur_fiber) = 1;
    %         tmp = imdilate(tmp, strel('disk', 4));
    %         one_node_mask(find(tmp)) = 0;
    %     end
    if match >= 1
        val = a_mask(cur_fiber(1));
        one_node_MBP_NEW(cur_fiber) = val;
    end
    
end
one_node_MBP = one_node_MBP_NEW;

%tmpLines = imdilate(one_node_MBP, strel('sphere', dil_amount_myelin));
tmpLines = one_node_MBP;
tmp_new_CASPR = new_CASPR;
for Y = 1:length(new_CASPR_idx)
    if tmpLines(new_CASPR_idx{Y}) < 1
        tmp_new_CASPR(new_CASPR_idx{Y}) = 0;
    end
end
one_node = one_node_MBP;
one_node_caspr = tmp_new_CASPR;
%one_node_mask = imdilate(one_node_mask, strel('disk', 2));
%coloc_sub_2 = imfuse(one_node_caspr, one_node_mask);
%figure(12); imshow(coloc_sub_2); title('> 1 internode');
figure(12); volshow(one_node_MBP, 'BackgroundColor', [0,0,0]);title('Subtracted nodes');


%% (3) Only keep the ones that match with 2 internodes
%new_CASPR_dil = imdilate(new_CASPR, strel('sphere', dil_amount_myelin));
new_CASPR_dil = new_CASPR;

%% TRY USING ENDPOINTS to eliminate new_internodes? - 28/04/2019
bw = one_node_MBP;
bw = imbinarize(bw);
skel = bwskel(bw);
endpoints = bwmorph3(skel, 'endpoints');
endpoints = imdilate(endpoints, strel('sphere', dil_amount_endpoint));
new_CASPR_dil(endpoints < 1) = 0; 
%

new_CASPR_dil = imdilate(new_CASPR_dil, strel('sphere', dil_amount_CASPR));
new_obj = bwconncomp(new_CASPR_dil);
new_CASPR_idx = new_obj.PixelIdxList;

two_node_mask = a_mask;
new_obj = bwconncomp(two_node_mask);
new_MBP_idx = new_obj.PixelIdxList;
two_node_mask_NEW = zeros(im_size);

%new_MBP_idx = a_locFibers;  % use locFibers
for T = 1:length(new_MBP_idx)
    cur_fiber = new_MBP_idx{T};
    match = 0;
    for Y = 1:length(new_CASPR_idx)
        cur_intnd = new_CASPR_idx{Y};
        same = ismember(cur_fiber, cur_intnd);
        if ~isempty(find(same, 1))
            match = match + 1;
        end
    end
%     if match < 2  % if not colocalized with 2, then delete
%         tmp = zeros(im_size);   % Dilate b/c locFibers is a skeleton
%         tmp(cur_fiber) = 1;
%         tmp = imdilate(tmp, strel('disk', 4));
%         two_node_mask(find(tmp)) = 0;
%     end
    if match >= 2
        val = a_mask(cur_fiber(1));
        two_node_mask_NEW(cur_fiber) = val;
    end
        
end
two_node_mask = two_node_mask_NEW;

%tmpLines = imdilate(two_node_mask, strel('sphere', dil_amount_myelin));
tmpLines = two_node_mask;
tmp_new_CASPR = new_CASPR;
for Y = 1:length(new_CASPR_idx)
    if tmpLines(new_CASPR_idx{Y}) < 1
        tmp_new_CASPR(new_CASPR_idx{Y}) = 0;
    end
end
two_nodes = two_node_mask;
two_nodes_caspr = tmp_new_CASPR;
%two_node_mask = imdilate(two_node_mask, strel('disk', 2));
%coloc_sub_3 = imfuse(two_nodes_caspr, two_node_mask);
%figure(13); imshow(coloc_sub_3); title('> 2 internodes');
figure(13); volshow(two_node_mask, 'BackgroundColor', [0,0,0]);title('Subtracted nodes');

%% Save results
name = erase(filename_raw, '.tif');

% cd(saveDirName);
% figure(9);
% filename = strcat('Result', num2str(fileNum_sav), '_', filename_raw, '_', '_Raw nodes and fibers.png');
% print(filename,'-dpng'); hold off;
% 
% figure(12);
% filename = strcat('Result', num2str(fileNum_sav), '_', filename_raw, '_', '_Single node.png') ;
% print(filename,'-dpng'); hold off;
% 
% figure(13);
% filename = strcat('Result', num2str(fileNum_sav), '_', filename_raw, '_', '_Double nodes.png') ;
% print(filename,'-dpng'); hold off;
% cd(cur_dir);

end


