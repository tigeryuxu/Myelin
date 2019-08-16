%% Need to add for 3D:
% 1) ridges2lines ==> needs to separate into 3 types of angles (x,y,z) ==>
% OR just take it out completely???
% 2) must fix rest of code to adapt to nanofiber cultures
% 3) must fix all "disk" dilations to "spheres"



% IF WANT TO ADJUST/Elim background, use ImageJ
% ==> 1) Split channels, 2) Select ROI, 3) go to "Edit/Clear outside"
% 4) Merge Channels, 5) Convert "Stack to RGB", 6) Save image

%***Note for Annick: ==> could also use ADAPTHISTEQ MBP for area at end...
%but too much???

% ***ADDED AN adapthisteq to imageAdjust.mat... 2019-01-24

%% Main function to run heuristic algorithm
opengl hardware;
close all;

cur_dir = pwd;

addpath(strcat(cur_dir, '\Cell_body_seg'))  % adds path to functions
addpath(strcat(cur_dir, '\Cell_counting'))  % adds path to functions
addpath(strcat(cur_dir, '\Demo-data'))  % adds path to functions
addpath(strcat(cur_dir, '\Image_enhance'))  % adds path to functions
addpath(strcat(cur_dir, '\IO_func'))  % adds path to functions
addpath(strcat(cur_dir, '\Ridge_filter'))  % adds path to functions
addpath(strcat(cur_dir, '\Skeletonize'))  % adds path to functions

cd(cur_dir);

foldername = uigetdir();   % get directory
moreTrials = 'Y';
trialNum = 1;

saveName = strcat(foldername, '_');
saveDirName = create_dir(cur_dir, saveName);   % creates directory
mkdir(saveDirName);

%% Run Analysis
cd(foldername);   % switch directories
fnames = dir('*.tif');

namecell=cell(1);
idx = 1;
for i=1:length(fnames)
    namecell{idx,1}=fnames(i).name;
    idx = idx + 1;
end
trialNames = namecell;
numfids = length(trialNames);   %%% divided by 5 b/c 5 files per pack currently
natfnames=natsort(trialNames);

%% Read in images
empty_file_idx_sub = 0;
for fileNum = 1 : 4: numfids
    
    cd(foldername);
    filename_raw = natfnames{fileNum + 1};
    channel_fibers = load_3D_gray(filename_raw);
    filename_raw = natfnames{fileNum + 2};
    channel_CASPR  = load_3D_gray(filename_raw);
    
    cd(cur_dir);

    mask = channel_fibers;
    greenImage = channel_CASPR;
    DAPImetric = 0.9
    hor_factor = 0
    minLength = 12 * 0.312
    name = ''
    %% Beginning of script from original heuristic algorithm:
    
    %internode_size = 50;  % BACKGROUND SUBTRACTION SIZE
    im_size = size(mask);
    internode_size = 0;
    DAPIsize = 0;
    dil_lines = 'N';
    enhance_DAPI = 'Y';
    fileNum_sav = 0;
    
    %% CAN COMMENT OUT NEXT 2 LINES
    %[s_overall, together, only_branched_fibers, final_no_sub_fibers] = line_traversal_3D(mask, im_size, minLength, dil_lines);
    %mask = final_no_sub_fibers;
    
    
    [all_internodes, all_caspr_coloc, one_node, one_node_caspr, two_nodes, two_nodes_caspr, bw_internd] = find_internodes_3D(greenImage, mask, DAPIsize, DAPImetric, enhance_DAPI, internode_size, im_size, hor_factor, minLength, dil_lines, cur_dir, saveDirName, filename_raw, fileNum_sav);
    %[all_internodes_b, all_caspr_coloc_b, one_node_b, one_node_caspr_b, two_nodes_b, two_nodes_caspr_b, bw_internd_b] = find_internodes_3D_branched(s_overall, greenImage, mask, DAPIsize, DAPImetric, enhance_DAPI, internode_size, im_size, hor_factor, minLength, dil_lines, cur_dir, saveDirName, filename_raw, fileNum_sav);
    all_internodes_b = 0,    all_caspr_coloc_b = 0,    one_node_b = 0;
    one_node_caspr_b = 0,     two_nodes_b = 0,    two_nodes_caspr_b = 0;
    bw_internd_b = 0;
    
    %% Calculate nodal distances - VERY SLOW CURRENTLY
    largest_distance = 5 % pixels
    %[all_nodal_dist] = get_nodal_distances_3D(all_caspr_coloc, largest_distance);
    %[one_caspr_nodal_dist] = get_nodal_distances_3D(one_node_caspr, largest_distance);
    %[two_caspr_nodal_dist] = get_nodal_distances_3D(two_nodes_caspr, largest_distance);
    
    %% Get actual undilated size of nodes from original MBP image
    bw_green = binarize_3D_otsu(greenImage);
    figure(300); volshow(bw_green);
    tmp = bw_green;
    bw_green(all_caspr_coloc < 1) = 0; all_caspr_coloc = bw_green; bw_green = tmp;
    bw_green(one_node_caspr < 1) = 0; one_node_caspr = bw_green; bw_green = tmp;
    bw_green(two_nodes_caspr < 1) = 0; two_nodes_caspr = bw_green; bw_green = tmp;
    
    % turn these into bwdist ==> to get nodal length!
    cd(cur_dir);
    cd(saveDirName);
    save_internode_data_3D(mask, saveDirName)
    save_internode_data_3D(all_internodes, saveDirName)
    save_internode_data_3D(all_internodes_b, saveDirName)

    save_internode_data_3D(all_caspr_coloc, saveDirName)
    save_internode_data_3D(all_caspr_coloc_b, saveDirName)

    save_internode_data_3D(one_node, saveDirName)
    save_internode_data_3D(one_node_b, saveDirName)

    save_internode_data_3D(one_node_caspr, saveDirName)
    save_internode_data_3D(one_node_caspr_b, saveDirName)

    save_internode_data_3D(two_nodes, saveDirName)
    save_internode_data_3D(two_nodes_b, saveDirName)

    save_internode_data_3D(two_nodes_caspr, saveDirName)
    save_internode_data_3D(two_nodes_caspr_b, saveDirName)
    
    L = ['-'];
    dlmwrite(strcat('internodes', saveDirName, '.csv'), L, '-append');
    
    %% Comment out nodal distances for now b/c algorithm too slow in 3D
    %if isempty(all_nodal_dist)  all_nodal_dist = 0;  end
    %dlmwrite(strcat('internodes', saveDirName, '.csv'), all_nodal_dist, '-append') ;
    
    %if isempty(one_caspr_nodal_dist)  one_caspr_nodal_dist = 0;  end
    %dlmwrite(strcat('internodes', saveDirName, '.csv'), one_caspr_nodal_dist, '-append') ;
    
    %if isempty(two_caspr_nodal_dist)  two_caspr_nodal_dist = 0;  end
    %dlmwrite(strcat('internodes', saveDirName, '.csv'), two_caspr_nodal_dist, '-append') ;
    
    
    %figure(5);
    %set(gcf, 'InvertHardCopy', 'off');   % prevents white printed things from turning black
    
    all_internodes = imadd(im2double(all_internodes), all_internodes_b);
    filename_save = strcat('Result', erase(name, '*'), num2str(fileNum_sav), '_', filename_raw, '_', '_(0) RAW_linear_objects.tif');
    save_3D_combine(mask, zeros(size(all_internodes)), zeros(size(all_internodes)), filename_save, im_size)
    
    all_caspr_coloc = imadd(all_caspr_coloc, all_caspr_coloc_b);
    filename_save = strcat('Result', erase(name, '*'), num2str(fileNum_sav), '_', filename_raw, '_', '_(4) All-raw-internodes.tif');
    save_3D_combine(all_internodes, all_caspr_coloc, zeros(size(all_internodes)), filename_save, im_size)
    
    one_node = imadd(one_node, one_node_b);
    one_node_caspr = imadd(one_node_caspr, one_node_caspr_b);
    filename_save = strcat('Result', erase(name, '*'), num2str(fileNum_sav), '_', filename_raw, '_', '_(3) one-node-coloc-internodes.tif');
    save_3D_combine(one_node, one_node_caspr, zeros(size(all_internodes)), filename_save, im_size)
    
    
    two_nodes = imadd(two_nodes, two_nodes_b);
    two_nodes_caspr = imadd(two_nodes_caspr, two_nodes_caspr_b);
    filename_save = strcat('Result', erase(name, '*'), num2str(fileNum_sav), '_', filename_raw, '_', '_(2) two-node-coloc-internodes.tif');
    save_3D_combine(two_nodes, two_nodes_caspr, zeros(size(all_internodes)), filename_save, im_size)
    
    %filename_save = strcat('Result', erase(name, '*'), num2str(fileNum_sav), '_', filename_raw, '_', '_(1) raw image.tif');
    %save_3D_combine(redImage, greenImage, zeros(size(all_internodes)), filename_save, im_size)
    
    
    %print(filename,'-dpng')
    %hold off;
    cd(cur_dir);
    close all;    
    
end

