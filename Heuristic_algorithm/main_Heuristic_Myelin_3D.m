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

%% Initialize
allWrappedR = []; allWrappedG = []; allTotalCells = [];
allNames = cell(0); allTrialLengths = cell(0);  allTrialSheathsR = cell(0); allTrialSheathsG = cell(0); allInfoInfo = cell(0);
allTrialS = cell(0); allTrialMeanFLC = cell(0);
allSumO4 = []; allSumMBP = [];

foldername = uigetdir();   % get directory
moreTrials = 'Y';
trialNum = 1;

saveName = strcat(foldername, '_');
saveDirName = create_dir(cur_dir, saveName);   % creates directory
mkdir(saveDirName);


%% Loads up GUI
% '10', '35', 'N', '25', '25', 'N', 'N', '0.25', '0.227', '0.8', '4', 'N', 'N', '75', 'N'
% '20', '55', 'N', '150', '300', 'N', 'Y', '0.6', '0.227', '0.9', '8', 'N', 'N', '200', 'N'
%save_params = {'', '', '0.227', '20', '8', '0.9', '55', '300', '0', '0', '0', '0', '0'};
%save_params = {'', '', '0.454', '10', '4', '0.9', '35', '25', '0', '0', '0', '0', '0'};
%human_OL_save_params = {'', '', '0.454', '10', '6', '1.5', '35', '25', '0', '0', '0', '0', '0'};
%rat_OL_AUTOMATED_params = {'', '', '0.454', '10', '6', '0.5', '35', '25', '0', '0', '0', '0', '0'};

[output] = GUI;
save_params = cell(0);
calibrate = 1;

while (calibrate)
    name = get((output.name), 'String');
    batch = get((output.batch), 'String');
    scale = str2double(get((output.scale), 'String'));
    diameterFiber = str2double(get((output.diameter_F), 'String'));
    sigma = str2double(get((output.sigma), 'String'));
    sensitivity = str2double(get((output.sensitivity), 'String'));
    minLength = str2double(get((output.minLength), 'String'))  / scale;
    DAPIsize = str2double(get((output.DAPIsize), 'String')) / (scale * scale);
    unscaled_DAPI = str2double(get((output.DAPIsize), 'String'));
    nanoYN = get((output.checkbox12), 'value');
    combineRG = get((output.Combine_RG), 'value');
    verbose = get((output.verbose), 'value');
    calibrate = get((output.calib), 'value');
    match_words = get((output.match_full_name), 'value');
    bool_load_five = get((output.checkbox11), 'value');
    adapt_his = get((output.Nano_YN), 'value');
    divide_im = get((output.checkbox13), 'value');
    hor_factor = get((output.checkbox16), 'value');
    switch_sheaths = get((output.checkbox17), 'value');
    
    save_params = {name, batch, scale, diameterFiber, sigma, sensitivity, minLength * scale, unscaled_DAPI, nanoYN...
        ,combineRG ,verbose, calibrate, match_words, bool_load_five, adapt_his, divide_im, hor_factor, switch_sheaths};
    
    if calibrate
        sensitivity =  calib(sigma, foldername, cur_dir);
        defaultans{10} = num2str(sensitivity);
    end
end
cd(saveDirName);
save('___Parameters used___', 'save_params');
cd(cur_dir);

%% TO ADD as GUI prompts:
%square_cut_h = 6700; % 2052 for 8028 images, and %1600 for newer QL images  and 1500 for HA751
%square_cut_w = 7200;

%% FOR DARYAN: human_OL == N, enhance_DAPI == N, size == 1000, switch_sheaths on line 564 == 1

% scale of LIF internodes == 0.3611

find_internode = 1;  % change to 1 if want to use internode analysis!!!
vol_bool = 1;
% FOR HUMAN TRIALS, need to eliminate more smaller cells???
enhance_RED = 'N';
human_OL = 'Y';
if switch_sheaths == 1
    human_OL = 'N';
end

if human_OL == 'Y'
    squareDist = 150;
    enhance_RED = 'Y'; % ==> set as 'Y' for other human OL trials!!!
else
    squareDist = 150;
    %squareDist = 50;
end



square_cut_h = 1460; % 2052 for 8028 images, and %1600 for newer QL images  and 1500 for HA751
square_cut_w = 1936;

remove_nets = 'Y'; % background subtraction for O4

mag = 'N';
enhance = 'Y';   % (Background subtraction O4 images) CLEM ==> doesn't need this
enhance_DAPI = 'Y'; % (Background subtraction O4 images) Daryan images w/ noise ==> doesn't need this


%% PRESET and scaled
DAPImetric = 0.3;   % Lower b/c some R not picked up due to shape...
percentDilate = 2;   % for cores
%hor_factor = 2;
near_join = round(3 / (scale));  % in um
fillHoles = round(8 / (scale * scale));  % in um^2
squareDist = round(squareDist / (scale));  % in um (is the height of the cell that must be obtained to be considered possible candidate)
coreMin = 0;
%elim_O4 = round(40 / (scale * scale));    % SMALL O4 body size (200 ==> 1000 pixels)

elim_O4 = round(5 / (scale * scale));    % SMALL O4 body size (200 ==> 1000 pixels)


DAPI_bb_size = round(unscaled_DAPI / (scale));

%% ADD TO MENU:
if bool_load_five == 1
    load_five = 5;
else
    load_five = 1;
end

if load_five == 5
    allChoices = choosedialog2();   %% read in choices  %%% SWITCH TO FROM GUI.m
end

batch_skip = 'Y';   % SHOULD BE ADDED TO GUI
batch_run = 'Y';
batch_num = 0;
batch = cell(1);   % intialize empty

% Clem:
%batch = {'Clem1-', 'Clem2-', 'Ctr'};
%save_params = {'', '', '0.454', '10', '4', '0.9', '35', '25', '0', '0', '0', '0', '0'};
%save_params = {'', '', '0.227', '20', '8', '0.9', '12', '8', '0', '0', '0', '0', '0'};

%%
%batch = {'*Olig2_WT', '*Olig2_KO'};
%batch = {'*KOSkap2_20x', '*WT_20x'};
%batch = {''};

%batch = {'*C1', '*C2', '*C3', '*RR1', '*RR2', '*RR3'};

%batch = {'n1_KO', 'n1_WT', 'n2_KO', 'n2_WT', 'n3_20xzoom_MBP_KO',  'n3_20xzoom_MBP_WT', 'n4_20x_zoom_KO', 'n4_20x_zoom_WT'};

%batch = {'n1_20x_KO', 'n1_20x_WT', 'n2_KOSkap2_20x', 'n2_WT_20x', 'n3_20x_snap_MBP_CD140_WT_', 'n3_20x_snap_MBP_CD140_KO_',  'n3_snap_20x_MBP_Olig2_KO_', 'n3_snap_20x_MBP_Olig2_WT_',   'n4_20x_MBP_KO', 'n4_20x_MBP_WT', 'n5_KO', 'n5_WT'};

%batch = {'12 wpg', '16 wpg'};

%% Run Analysis
batch_numFiles = [];
batch_sizes = [];
while (moreTrials == 'Y')
    
    %% Batch processing
    batch_num = batch_num + 1;
    if batch_skip == 'N'
        default = {'', 'N'};
        prompt = {'image set name:', 'Batch?'};
        dlg_title = 'Input';
        num_lines = 1;
        answer = inputdlg(prompt,dlg_title,num_lines,default);
        name = cell2mat(answer(1));
        batch_run = cell2mat(answer(2));
    end
    
    if batch_num > length(batch)  % ends everything
        break;
    end
    
    if batch_run == 'Y'
        batch_skip = 'Y';
        name = batch{batch_num};
    end
    
    %% Reads in all the files to analyze in the current directory
    [sumWrappedR, sumUnWrappedR, sumFibersPerPatchR, sumO4] = deal(0);   % variable declaration
    [sumWrappedG, sumUnWrappedG, sumFibersPerPatchG, sumMBP] = deal(0);
    allLengthFibers = [];    allLengthFibersG = [];    allNumSheathsR = [];    allNumSheathsG = []; allMeanLPC = [];
    allS = []; allInfo = [];
    
    cd(saveDirName);
    if isempty(name)
        name = '';   % Output file
    end
    nameTmp = strcat('allAnalysis', erase(name, '*'), '.txt');   % Output file
    
    fileID = fopen(nameTmp,'w');
    fid_internodes = fopen(strcat('internodes', saveDirName, '.csv'), 'w') ;
    
    cd(cur_dir);
    
    cd(foldername);   % switch directories
    nameCat = strcat(name, '*tif*');
    fnames = dir(nameCat);
    
    namecell=cell(1);
    idx = 1;
    for i=1:length(fnames)
        if  match_words && ~isstrprop(fnames(i).name(length(name) + 1), 'digit')   % if need to match full word, and it does NOT match, then skip
            continue;
        end
        namecell{idx,1}=fnames(i).name;
        idx = idx + 1;
    end
    trialNames = namecell;
    numfids = length(trialNames);   %%% divided by 5 b/c 5 files per pack currently
    
    batch_numFiles = [batch_numFiles, numfids];
    %% Read in images
    empty_file_idx_sub = 0;
    for fileNum = 1 : load_five: numfids
        
        cd(cur_dir);
        natfnames=natsort(trialNames);
        filename_raw = natfnames{fileNum};
        %% Decide if want to load individual channels or single image
        cd(foldername);
        [redImage, greenImage, DAPIimage] = load_3D(filename_raw);
         
        %% Subtract background for internode data
        if find_internode == 1
            min_MBP_intensity = 10/255;
            I = imgaussfilt3(redImage, 0.5);
            background = imopen(I,strel('disk',30));
            I2 = imsubtract(I, background);
            I = I2;
            I(I < min_MBP_intensity) =  0;
            redImage = I;
            
            % for green channel
            I = imgaussfilt3(greenImage, 0.5);
            background = imopen(I,strel('disk',20));
            I2 = imsubtract(I, background);
            I = I2;
            
            %% SUBTRACTS OUT ANY INTENSITY BELOW 80 in GREEN CHANNEL (for internodes)
            % used 60 for first run
            min_internodes_intensity = 60/255;
            I(I < min_internodes_intensity) = 0;
            greenImage = I;
            
        end
        
        if load_five == 5 && fileNum > 1
            fileNum_sav = ((fileNum - 1)/load_five) + 1;
        else
            fileNum_sav = (fileNum);
        end
        
        intensityValueDAPI = DAPIimage;
        O4_original = redImage;
        MBP_im = greenImage;
        
        %% 2018-11-13: MADE THE BACKGROUND SUBTRACTION LARGER so won't lose pieces of cell in analysis
        O4_im_ridges = O4_original;
        
        %% 2018-12-14: ADDED??? b/c should be looking at ridges from the actual original image??? not binarized???
        O4_im_ridges_adapted = O4_original;
        
        
        [O4_im, originalRed] = imageAdjust_3D(O4_original, fillHoles, enhance);   % image adjust
        
        if enhance_RED == 'Y'
            O4_im = imclose(O4_im, strel('disk', 2));
        end
        
        
        %% 2018-12-14: Can remove the weird membrane between sheaths
        if remove_nets == 'Y'
            tmp_im = O4_original;
            background = imopen(O4_original,strel('disk',10));
            I2 = imsubtract(O4_original, background);
            I = I2;
            O4_im_ridges_adapted = tmp_im;
            %I = adapthisteq(I);
        end
        
        % Combine
        if combineRG == 0
            combined_im = O4_im;
        elseif combineRG == 1
            combined_im = imbinarize(O4_im + MBP_im);
        end
        isGreen = 0;
        figure(1); volshow(combined_im);
        
        siz = size(combined_im);
        
        %% (1) Find peaks for DAPI
        %DAPIsize = 10;
        if find_internode == 1
            intensityValueDAPI = greenImage;
            %enhance_DAPI = 'N';
        end
        %% TIGER - CHANGED "enhance", "Human_OL", and "cropping size" ==> all for Daryan's stuff
        [mat, objDAPI, DAPI_bw] = DAPIcount_3D(intensityValueDAPI, DAPIsize, DAPImetric, enhance_DAPI, DAPI_bb_size);  % function
        
        if length(objDAPI) > 50000
            continue;
        end
        
        
        %% Initializes struct to store everything
        c= cell(length(objDAPI), 1); % initializes Bool_W with all zeros
        [c{:}] = deal(0);
        strucMat = num2cell(mat, 2);
        s = struct('objDAPI', objDAPI', 'centerDAPI', strucMat, 'Core', cell(length(objDAPI), 1)...
            ,'CB', cell(length(objDAPI), 1), 'Fibers', cell(length(objDAPI), 1), 'Mean_Fiber_L_per_C', cell(length(objDAPI), 1), 'Bool_W', c...
            , 'im_num', c, 'O4_bool', c, 'AreaOverall', c, 'numO4', c, 'im_size', c, 'OtherStats', c);
        
        
        %% Added "size" metric to keep track of size of image - Tiger 16/02/2019
        size_red = size(intensityValueDAPI);
        %redImage = redImage(1:square_cut_h, 1: square_cut_w, :);
        height = size_red(1);
        width = size_red(2);
        
        %% ONLY DO THESE IF NO INTERNODES TO FIND
        if find_internode == 0
      
            s(1).im_size = [height, width];

            %% Store the size of images for batch processing - Tiger 12/02/19
            if fileNum == 1 || length(batch_numFiles) == 1
                batch_sizes = [batch_sizes; [height, width]];
            end

            %% (2) Extract cores
            [s] = reg_core_filt_3D(combined_im, diameterFiber, siz, percentDilate, s);  % small cores
            %^^^TAKES A LONG TIME - Tiger Xu - 06/03/2019, tried to
            %optimize

            [cb, no_dilate_cb, s] = cell_body_filt_3D(combined_im, diameterFiber, siz, coreMin, s);        % cell body

            % Updates "cores"
            cores = zeros(siz);
            for Y = 1:length({s.Core})
                if ~isempty(s(Y).Core)
                    cores(s(Y).Core) = 1;
                end
            end

            [cores, cb, s] = match_cores(cores, cb, siz, s);  % match_cores
            if verbose
                figure(71); imshow(cores); title('Associated Cores');
            end

            %% (3) Count O4+ cells:
            [cores, cb, unass_cb, s] = O4_count_3D(combined_im, cores, cb, siz,diameterFiber, elim_O4, s);

            if verbose
                figure(70); imshow(unass_cb); title('Unassociated Objects');
            end

            %% Print * for DAPI and O4+
            if enhance_RED == 'Y'
                tmpDAPI = histeq(intensityValueDAPI);
                O4_tmp = histeq(O4_original);
            elseif switch_sheaths == 1
                O4_tmp = O4_im_ridges_adapted;
                tmpDAPI = intensityValueDAPI;

            else
                tmpDAPI = histeq(intensityValueDAPI);
                O4_tmp = histeq(O4_im_ridges_adapted);

            end

            %% Switch the sheaths for Annick's analysis
            greenOrig = MBP_im;
            if switch_sheaths == 1
                MBP_im = MBP_im;
            else
                MBP_im = zeros(size(O4_im));
            end

            if combineRG == 1
                MBP_im = imadjust(greenOrig);
            end

            if find_internode == 0   % only do this if NOT find_internodes
                wholeImage = cat(3, O4_tmp, MBP_im, tmpDAPI);
            else
                wholeImage = cat(3, O4_tmp, histeq(greenImage), DAPIimage);
            end

            %% Switch the sheaths for Annick's analysis
            if switch_sheaths == 1
                %O4_im_ridges_adapted = MBP_im;
                O4_im_ridges_adapted = greenOrig;
            end

            figure(5); volshow(wholeImage); title('Output Image'); hold on;
            for Y = 1:length({s.objDAPI})
                if ~isempty(s(Y).centerDAPI) % Print DAPI
                    text(s(Y).centerDAPI(1, 1),  s(Y).centerDAPI(1 ,2), s(Y).centerDAPI(1 ,3), '^',  'color','m' ,'Fontsize',8);   % writes "peak" besides everything
                end

                if ~isempty(s(Y).Core)  % Print O4+
                    text(s(Y).centerDAPI(1, 1),  s(Y).centerDAPI(1 ,2), s(Y).centerDAPI(1 ,3), 'O4',  'color','y' ,'Fontsize',5);   % writes "peak" besides everything
                    sumO4 = sumO4 + 1;
                    s(Y).O4_bool = 1;
                end
            end
       
            %% (4) Sort through and find cell objects that are much too small, and set them permanently to NOT wrapped
            new_combined_im = binarize_3D_otsu(imadd(unass_cb, combined_im));     % new combined_im also must include the dilated image in O4_count
            
            %% 2018-11-13: REMOVED A SETTING FOR HUMAN OLs, now will allow OLs that look like thin lines
            if switch_sheaths
                s = s;
            else
                [s] = small_del_O4(new_combined_im, minLength, squareDist, siz, s);
            end
        
        end
        
        %% (5) Line seg:
        % 2019-01-17: for Annick Baron, sens = 10, sigma = 1
        if adapt_his
            O4_im_ridges_adapted = histeq(O4_im_ridges_adapted);
        end
        [fibers, fibers_idx, Lambda2R] = line_seg_3D(O4_im_ridges_adapted, sigma, sensitivity);
        figure(900); volshow(fibers, 'BackgroundColor', [0,0,0]);

        %% (6) Clean fibers by subtracting out CB
        %% 19/01/24 - Tiger added: don't sub cell bodies for Annick
        if switch_sheaths == 0 && find_internode == 0
            fibers = binarize_3D_otsu(imsubtract(fibers, cb));
            if mag == 'Y'
                fibers = imopen(fibers, strel('disk', 2));   % to get rid of straggling thin strands
            end
        end
        
        %% (7) NEW LINE ANALYSIS (transforms ridges to lines)
        % Horizontal lines are more dim
        dil_lines = 'Y';
        if scale > 0.3
            dil_lines = 'N';
        end
        
        if width > 5500  % if the image is very large, then also dilate the fibers
            dil_lines = 'Y';
        end
        
        [all_lines, locFibers,allLengths, mask, fibers] = ridges2lines_3D(fibers, siz, hor_factor, minLength, dil_lines);
        figure(1000); volshow(mask, 'BackgroundColor', [0,0,0]);
        
        
        %% TIGER - can insert internode analysis here
        if find_internode == 1
            %internode_size = 50;  % BACKGROUND SUBTRACTION SIZE
            im_size = size(mask);
            internode_size = 5;
            DAPIsize = 5;
            dil_lines = 'N';
            enhance_DAPI = 'Y';
            [all_internodes, all_caspr_coloc, one_node, one_node_caspr, two_nodes, two_nodes_caspr, bw_internd] = find_internodes_3D(greenImage, mask, DAPIsize, DAPImetric, enhance_DAPI, internode_size, im_size, hor_factor, minLength, dil_lines, cur_dir, saveDirName, filename_raw, fileNum_sav);
            
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
            save_internode_data_3D(all_caspr_coloc, saveDirName)
            save_internode_data_3D(one_node, saveDirName)
            save_internode_data_3D(one_node_caspr, saveDirName)
            save_internode_data_3D(two_nodes, saveDirName)
            save_internode_data_3D(two_nodes_caspr, saveDirName)
            
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
            filename_save = strcat('Result', erase(name, '*'), num2str(fileNum_sav), '_', filename_raw, '_', '_(0) RAW_linear_objects.tif');
            save_3D_combine(mask, zeros(size(all_internodes)), zeros(size(all_internodes)), filename_save, im_size)
            
            filename_save = strcat('Result', erase(name, '*'), num2str(fileNum_sav), '_', filename_raw, '_', '_(4) All-raw-internodes.tif');
            save_3D_combine(all_internodes, all_caspr_coloc, zeros(size(all_internodes)), filename_save, im_size)
            
            filename_save = strcat('Result', erase(name, '*'), num2str(fileNum_sav), '_', filename_raw, '_', '_(3) one-node-coloc-internodes.tif');
            save_3D_combine(one_node, one_node_caspr, zeros(size(all_internodes)), filename_save, im_size)
            
            
            filename_save = strcat('Result', erase(name, '*'), num2str(fileNum_sav), '_', filename_raw, '_', '_(2) two-node-coloc-internodes.tif');
            save_3D_combine(two_nodes, two_nodes_caspr, zeros(size(all_internodes)), filename_save, im_size)
            
            filename_save = strcat('Result', erase(name, '*'), num2str(fileNum_sav), '_', filename_raw, '_', '_(1) raw image.tif');
            save_3D_combine(redImage, greenImage, zeros(size(all_internodes)), filename_save, im_size)
            
            
            %print(filename,'-dpng')
            %hold off;
            cd(cur_dir);
            close all;
            continue;
            
            
        end
        
        locFibers =  locFibers(~cellfun('isempty',locFibers));   % delete from the list if not a line
        allLengths = allLengths(~cellfun('isempty', allLengths));   % delete from the list if not a line
        
        %% (8) Check CBs to see if wrapped or not
        fibers_sub_cb = bwmorph(cb, 'thicken', 3);
        [locFibers, allLengths, s] = wrappingAnalysis(fibers_sub_cb, locFibers, allLengths, siz, minLength, isGreen, s);
        
        %% (9) Check remaining fibers with fibers_sub_cb (real), to see if wrapped or not
        % COUNT AGAIN, with a FULL fibers_sub_cb, to get all the fibers NOT directly connected to stuff
        % and use only the REMAINING fibers (i.e. fibers(locFibers) = 0) ==> set to zero the already found ones
        
        if switch_sheaths == 1
            
            O4_adapt = adapthisteq(O4_original);
            [combined_im, originalRed] = imageAdjust(O4_adapt, fillHoles, enhance);
            
            
            %% Alternative is to have this less filled in image (more holes)
            %background = imopen(O4_original,strel('disk',150));
            %I2 = I - background;
            %I = I2;
            %combined_im = imbinarize(adapthisteq(I));
            
            %% MAYBE CAN ADD WATERSHED HERE??? using the CBs as minima??? So create discrete cells????
            %% AND THEN GET THE AREA FROM HERE AS WELL???
            %% ALSO USE THIS TO DELINEATE CBs better???
            
            bw = ~bwareaopen(~combined_im, 10);  % clean
            D = -bwdist(~bw);  % EDT
            D2 = imimposemin(D, cb);
            
            Ld2 = watershed(D2);
            bw3 = bw;
            bw3(Ld2 == 0) = 0;
            bw = bw3;
            
            figure(119); title('CB watershed');
            [B,L] = bwboundaries(bw, 'noholes');
            imshow(bw);
            imshow(label2rgb(L, @jet, [.5 .5 .5]));
            hold on;
            
            
            %% Loop through and only keep watershed CBs that coloc with "cb" mask
            
            
            %% Then pick out MBP area WITHIN each CB identified to find AREA of each cell!!!
            
            %% Now okay to do rest of analysis below b/c watershed left spaces between CBs
            
            combined_im = bw;
        end
        
        fibers_sub_cb = imbinarize(combined_im - cb);  % THE REAL FIBERS_sub_cb
        [sub_locFibers, allLengths, s] = wrappingAnalysis(fibers_sub_cb, locFibers, allLengths, siz, minLength, isGreen, s);
        
        locFibers = sub_locFibers; % THESE ARE THE FIBERS STILL UN-ASSOCIATED
        
        %% (10) Associate the remaining lines with the nearest DAPI point
        [locFibers, s, bw_final_fibers] = near_line_join(locFibers, near_join, siz, verbose, s);
        
        %% SAVE RESULTS:
        idx_wrap = find([s.Bool_W] == 1);
        numWrappedR = length(idx_wrap); % set numWRAPPED
        
        % Count allNumSheaths and allLengths  ***using LENGTH OF SKELETON
        for N = 1:length({s.objDAPI})
            if s(N).Bool_W == 1
                allNumSheathsR = [allNumSheathsR length(s(N).Fibers)];
                
                fibers_cell = [];
                for Y = 1:length(s(N).Fibers)
                    allLengthFibers = [allLengthFibers length(s(N).Fibers{Y})];
                    fibers_cell = [fibers_cell length(s(N).Fibers{Y})];
                end
                
                avg_length = mean(fibers_cell) * scale;
                s(N).Mean_Fiber_L_per_C = avg_length;  % add to struct
                
                allMeanLPC = [allMeanLPC avg_length];
            end
        end
        
        % Save figures
        wrappedDAPIR = cell(0);  % to be saved for accuracy_eval
        for Y = 1:length(idx_wrap)
            wrappedDAPIR{end + 1} = s(idx_wrap(Y)).objDAPI;
        end
        
        if verbose
            saveFigs(saveDirName, cur_dir, name, fileNum, objDAPI, wrappedDAPIR);
        end
        
        numCells = length(objDAPI);   % Don't comment out
        
        numFibersG = 0;
        wrappedG = 0;
        unwrappedG = 0;
        
        lineNum = 1;
        allxy_long = cell(1,1);
        numFibersR = length(allLengthFibers);
        
        wrappedR = numWrappedR;
        unwrappedR = numCells - wrappedR;
        
        % Save image number as well:
        image_number = (fileNum);
        for Y = 1:length(s)
            s(Y).im_num = image_number;
        end
        
        
        
        %% ALSO get the MBP area ==> also need to colocalize the identified sheaths with original
        % (a) want the area of MBP overall in the whole image (can
        % normalize later to # of cells)
        % (b) want the area of MBP colocalized with identified
        % sheaths ==> use Julia's algo
        [bw_green, originalGreen] = imageAdjust(MBP_im, fillHoles, enhance);   % image adjust
        s(1).AreaOverall = nnz(bw_green); % gets area
        s(1).numO4 = sumO4;
        
        %% ALSO GET the intensity, width ect... within all the sheaths
        for N = 1:length({s.objDAPI})
            if s(N).Bool_W == 1
                fibers_cell = [];
                s(N).OtherStats = cell(0);
                for Y = 1:length(s(N).Fibers)
                    tmp_ridges = imbinarize(bw_final_fibers);
                    tmp_fibers = zeros(size(bw_final_fibers));
                    tmp_fibers(s(N).Fibers{Y}) = 1;
                    tmp_fibers = imdilate(tmp_fibers, strel('disk', 4));
                    tmp_ridges(tmp_fibers == 0) = 0;
                    stats = regionprops(tmp_ridges,MBP_im,'MeanIntensity', 'Area', 'Perimeter', 'MinorAxisLength', 'PixelValues');
                    if length(stats) > 1
                        [val, idx] = max([stats(:).Area]);
                        s(N).OtherStats{end + 1} = stats(idx);
                    else
                        s(N).OtherStats{end + 1} = stats;
                    end
                end
            end
        end
        
        %% For Annick's analysis and to find area of MBP when combine_RG == 1
        if switch_sheaths == 1 || Combine_RG == 1
            get_ensheathed_only = 1
            [cores_MBP, s] = find_MBP(s, O4_adapt, switch_sheaths, O4_original, MBP_im, fillHoles, enhance, greenOrig, get_ensheathed_only, zeros(size(O4_adapt)));
            get_ensheathed_only = 0
            subtract_old_MBP = cores_MBP;
            [cores_MBP, s] = find_MBP(s, O4_adapt, switch_sheaths, O4_original, cores_MBP, fillHoles, enhance, greenOrig, get_ensheathed_only, subtract_old_MBP);
            
        end
        
        %% Print images of results
        filename_raw = erase(filename_raw, '.tif');
        cd(saveDirName);
        figure(5);
        set(gcf, 'InvertHardCopy', 'off');   % prevents white printed things from turning black
        filename = strcat('Result', erase(name, '*'), num2str(fileNum_sav), '_', filename_raw, '_', num2str(counter), '1) All channels');
        print(filename,'-dpng')
        hold off;
        
        figure(1);
        filename = strcat('Result', erase(name, '*'), num2str(fileNum_sav),  '_', filename_raw, '_', num2str(counter), '2) Cell body BW') ;
        print(filename,'-dpng')
        hold off;
        
        % Saves image
        figure(100);
        filename = strcat('Result', erase(name, '*'), num2str(fileNum_sav),  '_', filename_raw, '_', num2str(counter), '3) Cell nuclei') ;
        print(filename,'-dpng')
        hold off;
        
        figure(31);
        filename = strcat('Result', erase(name, '*'), num2str(fileNum_sav),  '_', filename_raw, '_', num2str(counter), '4) final_fibers') ;
        print(filename,'-dpng')
        hold off;
        
        bw_final_fibers(bw_final_fibers > 0) = 1;
        
        figure(67); imshowpair(wholeImage, mask); title('Filter ridges');
        filename = strcat('Result', erase(name, '*'), num2str(fileNum_sav),  '_', filename_raw, '_', num2str(counter), '5) Filter ridges') ;
        print(filename,'-dpng'); hold off;
        
        figure(188); imshow(cat(3, zeros(size(MBP_im)), greenOrig,  zeros(size(MBP_im))));
        filename = strcat('Result', erase(name, '*'), num2str(fileNum_sav),  '_', filename_raw, '_', num2str(counter), '6) MBP alone') ;
        print(filename,'-dpng')
        hold off;
        
        figure(189); imshow(cat(3, O4_original, zeros(size(MBP_im)),  zeros(size(MBP_im))));
        filename = strcat('Result', erase(name, '*'), num2str(fileNum_sav),  '_', filename_raw, '_', num2str(counter), '7) Cell Body alone') ;
        print(filename,'-dpng')
        hold off;
        
        figure(88); imshowpair(wholeImage, imbinarize(bw_final_fibers)); title('Ridges to lines after sub core');  hold on;
        filename = strcat('Result', erase(name, '*'), num2str(fileNum_sav),  '_', filename_raw, '_', num2str(counter), '8) Skeletonized ridges') ;
        print(filename,'-dpng'); hold off;
        
        figure(32); imshow(bw_green);
        filename = strcat('Result', erase(name, '*'), num2str(fileNum_sav),  '_', filename_raw, '_', num2str(counter), '9) MBP bw') ;
        print(filename,'-dpng')
        hold off;
        
        if switch_sheaths || Combine_RG
            figure(121);
            filename = strcat('Result', erase(name, '*'), num2str(fileNum_sav),  '_', filename_raw, '_', num2str(counter), '10) CB watershed ensheathed') ;
            print(filename,'-dpng')
            hold off;
            
            figure(122);
            filename = strcat('Result', erase(name, '*'), num2str(fileNum_sav),  '_', filename_raw, '_', num2str(counter), '11) Ensheathed cores') ;
            print(filename,'-dpng')
            hold off;
            
            figure(123);
            filename = strcat('Result', erase(name, '*'), num2str(fileNum_sav),  '_', filename_raw, '_', num2str(counter), '12) MBP per cell') ;
            print(filename,'-dpng')
            hold off;
            
            figure(124);
            filename = strcat('Result', erase(name, '*'), num2str(fileNum_sav),  '_', filename_raw, '_', num2str(counter), '13) Non-ensheathed MBP cores') ;
            print(filename,'-dpng')
            hold off;
            
            figure(125);
            filename = strcat('Result', erase(name, '*'), num2str(fileNum_sav),  '_', filename_raw, '_', num2str(counter), '14) MBP per non-ensheathed cell') ;
            print(filename,'-dpng')
            hold off;
            
        end
        
        
        
        %% Print to file:
        %(1) "allAnalysis.txt" is for EACH image
        
        %%if file is already open, then no need to open it again
        fprintf(fileID,'Image No. : %d              File name: %s \n', (fileNum_sav), erase(name, '*'));
        fprintf(fileID,'Num wrapped R cells: %d \n', wrappedR);
        fprintf(fileID,'Num un-wrapped R cells: %d \n', unwrappedR);
        fprintf(fileID,'Proportion wrapped R: %.2f \n\n', wrappedR/(unwrappedR + wrappedR));
        fprintf(fileID,'Num wrapped G cells: %d \n', wrappedG);
        fprintf(fileID,'Num un-wrapped G cells: %d \n', unwrappedG);
        fprintf(fileID,'Proportion wrapped G: %.2f \n\n', wrappedG/(unwrappedG + wrappedG));
        
        fprintf(fileID,'Total num cells: %d \n', wrappedR + unwrappedR);
        fprintf(fileID,'Num R fibers in image: %d \n', numFibersR);
        fprintf(fileID,'Num G fibers in image: %d \n', numFibersG);
        
        fprintf(fileID,'length of wrapping per R fiber: %f', mean2(allLengthFibers));
        fprintf(fileID,'length of wrapping per G fiber: %f \n\n\n', mean2(allLengthFibersG));
        
        sumWrappedR = sumWrappedR + wrappedR;
        sumUnWrappedR = sumUnWrappedR + unwrappedR;
        sumFibersPerPatchR = sumFibersPerPatchR + numFibersR;
        
        sumWrappedG = sumWrappedG + wrappedG;
        sumUnWrappedG = sumUnWrappedG + unwrappedG;
        sumFibersPerPatchG = sumFibersPerPatchG + numFibersG;
        
        allInfo = [allInfo ; [wrappedR, numCells]];
        
        allS = [allS; s];
        
        cd(cur_dir);
        
        % Clears all figures
        arrayfun(@cla,findall(0,'type','axes'));
        
        counter = counter + 1;
        
        cd(saveDirName);
        
        save(strcat(erase(name, '*'), '_', num2str(fileNum_sav)), 'allS');
        
        %% Tiger: 20/01/19 - should add switch here so if batching, doesn't clear "allS"
        if length(batch) > 1
            allS = allS;   % doesn't delete it b/c it's batching
        else
            allS = [];
        end
        cd(cur_dir);
        
    end
    fclose(fileID);      %%%close after so it can append, but next time it writes file, it will over-write    %%% in the future maybe just append??? and have hours log
    
    cd(saveDirName);
    %% (2) "Summary.txt" is for summary of ALL the images
    proportionR = sumWrappedR/(sumUnWrappedR + sumWrappedR);
    proportionG = sumWrappedG/(sumUnWrappedG + sumWrappedG);
    
    nameTmp = strcat('summary', erase(name, '*'), '.txt');
    fileID = fopen(nameTmp,'w');
    fprintf(fileID,'Total num images analyzed: %d \n', numfids/5);
    fprintf(fileID,'Num wrapped R cells: %d \n', sumWrappedR);
    fprintf(fileID,'Num un-wrapped R cells: %d \n', sumUnWrappedR);
    fprintf(fileID,'Proportion wrapped R: %.2f \n', proportionR);
    
    fprintf(fileID,'Num wrapped G cells: %d \n', sumWrappedG);
    fprintf(fileID,'Num un-wrapped G cells: %d \n', sumUnWrappedG);
    fprintf(fileID,'Proportion wrapped G: %.2f \n\n', proportionG);
    
    fprintf(fileID,'Proportion Wrapped / O4+ cells: %.2f \n', sumWrappedR/ sumO4);
    fprintf(fileID,'Proportion O4+ / Total cells: %.2f \n', sumO4 / (sumWrappedR + sumUnWrappedR));
    fprintf(fileID,'Total num O4+ cells: %d \n', sumO4);
    fprintf(fileID,'Total num cells: %d \n', sumWrappedR + sumUnWrappedR);
    
    fprintf(fileID,'Total num R Fibers: %d \n', sumFibersPerPatchR);
    fprintf(fileID,'Avg length of wrapping per R fiber: %f \n\n', mean2(allLengthFibers));
    
    fprintf(fileID,'Total num G Fibers: %d \n', sumFibersPerPatchG);
    fprintf(fileID,'Avg length of wrapping per G fiber: %f \n', mean2(allLengthFibersG));
    
    fprintf(fileID,'Sensitivity of line segmentation: %.2f \n', sensitivity);
    
    fprintf(fileID,'User selected parameters %s -', save_params{:});
    fclose(fileID);
    
    cd(cur_dir);
    
    %% For stats later:
    allWrappedR = [allWrappedR sumWrappedR];
    allWrappedG = [allWrappedG sumWrappedG];
    allTotalCells = [allTotalCells (sumWrappedR + sumUnWrappedR)];
    allNames{trialNum} = name;
    allTrialLengths{trialNum} = allLengthFibers * scale;   %% SCALED values are saved
    allTrialSheathsR{trialNum} = allNumSheathsR;
    allTrialSheathsG{trialNum} = allNumSheathsG;
    
    allTrialMeanFLC{trialNum} = allMeanLPC;
    allTrialS{trialNum} = allS;
    
    allSumO4 = [allSumO4 sumO4];
    allSumMBP = [allSumMBP sumMBP];
    
    allInfoInfo{trialNum} = allInfo;
    
    %% Prompt if want to re-start the analysis
    if batch_run == 'Y'
        moreTrials = 'Y';
        trialNum = trialNum + 1;
    end
    
    %     if batch_run == 'N'
    %         questTitle='More Samples?';
    %         start(timer('StartDelay',1,'TimerFcn',@(o,e)set(findall(0,'Tag',questTitle),'WindowStyle','normal')));
    %         button2 = questdlg('Another analysis?', questTitle, 'Y','N','Y','Y');
    %         moreTrials = button2;
    %
    %     end
    %     trialNum = trialNum + 1;
    
end



%% Main function to run heuristic algorithm
% cur_dir = pwd;
% addpath(strcat(cur_dir, '\IO_func'))  % adds path to functions
% cd(cur_dir);
% saveDirName = uigetdir();   % get directory
% batch_numFiles = []; minLength = 12;
% scale = 0.454;

%% Combine individual *mat files to single .csvs
cd(cur_dir)
cd(saveDirName)
nameCat = '*mat';
fnames = dir(nameCat);
namecell=cell(1);
idx = 1;
for i=1:length(fnames)
    namecell{idx,1}=fnames(i).name;
    idx = idx + 1;
end
trialNames = namecell;
cd(cur_dir);
trialNames = natsort(trialNames);
cd(saveDirName);
numfids = length(trialNames) - 1;   %%% divided by 5 b/c 5 files per pack currently

%36, 44, 91, 93
% allNumSheathsR = [];
% allLengthFibersR = [];
% allMeanLPC = [];
% allLog = [];

%width = 6700;   % both should be 2052  or 1600
%height = 7200;  % 7729 x 5558

%7345 x 6898

%scale = 0.6904;
%scale = 0.454;
%minLengthSingle = 52;
minLengthSingle = 0;
minSingle = 12;   % in MICRONS
numO4 = 0;


all_individual_trials = [];
all_individual_trials_sheaths = cell(0);
all_individual_trials_lengths = cell(0);
all_individual_trials_log = cell(0);
all_individual_trials_LPC = cell(0);
all_individual_trials_area_per_cell = cell(0);
all_individual_trials_area_per_cell_non_ensheathed = cell(0);

batch_sort = 0;
batch_counter = 0;
size_counter = 1;
if length(batch_numFiles) > 1
    batch_sort = length(batch_numFiles);
    height = batch_sizes(size_counter, 1); % picks new size
    width = batch_sizes(size_counter, 2);
    batch_counter = batch_numFiles(size_counter);
    size_counter = size_counter + 1;
end


for fileNum = 1 : numfids
    
    allNumSheathsR = [];
    allLengthFibersR = [];
    allMeanLPC = [];
    allLog = [];
    all_area_per_cell = [];
    all_area_per_cell_non_ensheathed = [];
    num_O4_individual = 0;
    num_sheaths_individual = 0;
    
    filename = trialNames{fileNum};
    s = load(filename);
    s = s.allS;
    
    if isempty(s)
        all_individual_trials = [all_individual_trials; [0,  num_O4_individual,num_sheaths_individual]];
    else
        
        size_im = s(1).im_size;
        height = size_im(1);
        width = size_im(2);
        
        %% use diff sizes taken from the batch-runs if batched - Tiger 13/02/19
        if batch_counter == fileNum - 1 && batch_counter < numfids
            %height = batch_sizes(size_counter, 1); % picks new size
            %width = batch_sizes(size_counter, 2);
            %batch_counter = batch_counter + batch_numFiles(size_counter);
            %batch_counter = batch_counter + 1;
            %size_counter = size_counter + 1;
            %figure; imshow(tmp);
        end
        
        % Count allNumSheaths and allLengths  ***using LENGTH OF SKELETON
        
        all_area_per_cell = s(1).AreaOverall;
        all_area_per_cell_non_ensheathed = s(2).AreaOverall;
        for N = 1:length({s.objDAPI})
            
            if isfield(s, 'O4_bool') && s(N).O4_bool
                numO4 = numO4 + 1;
                num_O4_individual = num_O4_individual + 1;
            end
            
            if s(N).Bool_W == 1
                % Put the fibers into a tmp array so can find MajorAxisLength
                fibers_cell = [];
                tmp = zeros([height, width]);
                for Y = 1:length(s(N).Fibers)
                    tmp(s(N).Fibers{Y}) = 1;
                end
                
                
                %% For debug
                %figure(1); imshow(tmp);
                
                %tmp = imdilate(tmp, strel('disk', 10));
                [B,L] = bwboundaries(tmp, 'noholes');
                vv = regionprops(L, 'MajorAxisLength');
                num_sheaths = 0;
                
                new_vv = cell(0);
                for Y = 1:length(vv)
                    len = vv(Y).MajorAxisLength;
                    if len > minSingle / scale
                        new_vv{end + 1} = len;
                    end
                end
                vv = new_vv;
                
                %% EXTRA ==> eliminate single fibers + eliminate fibers that have double
                % where single length == 150 and double == 150/2
                
                if length(vv) == 1 && vv{1} < minLengthSingle / scale
                    continue;
                end
                
                if length(vv) == 2 && vv{1} < minLengthSingle /scale && vv{2} < minLengthSingle / scale
                    continue;
                end
                
                for Y = 1:length(vv)
                    len = vv{Y};
                    if len > minSingle / scale
                        allLengthFibersR = [allLengthFibersR len * scale];
                        fibers_cell = [fibers_cell len * scale];
                        log_length = log10(len * scale);
                        allLog = [allLog log_length];
                        num_sheaths = num_sheaths + 1;
                    end
                end
                
                if num_sheaths == 0
                    continue
                end
                allNumSheathsR = [allNumSheathsR num_sheaths];
                
                
                avg_length = mean(fibers_cell);
                %s(N).Mean_Fiber_L_per_C = avg_length;  % add to struct
                
                allMeanLPC = [allMeanLPC avg_length];
                
                num_sheaths_individual = num_sheaths_individual + 1;
                
            end
        end
        all_individual_trials = [all_individual_trials; [length({s.objDAPI}), num_O4_individual,num_sheaths_individual]];
    end
    num_sheaths_individual
    all_individual_trials_sheaths{end + 1} = allNumSheathsR;
    all_individual_trials_lengths{end + 1} = allLengthFibersR;
    all_individual_trials_log{end + 1} =  allLog;
    all_individual_trials_LPC{end + 1} = allMeanLPC;
    all_individual_trials_area_per_cell{end + 1} = all_area_per_cell;
    all_individual_trials_area_per_cell_non_ensheathed{end + 1} = all_area_per_cell_non_ensheathed;
end

cd(cur_dir);

% SAVE CSV FOR ALL INDIVIDUAL TRIALS
cd(saveDirName);
%csvwrite('output_props.csv', all_individual_trials);
split_name = strsplit(saveDirName, '\');
saveDirName = split_name{end};

fid1 = fopen(strcat('output_sheaths_', saveDirName, '.csv'), 'w') ;
fid2 = fopen(strcat('output_lengths_', saveDirName, '.csv'), 'w') ;
fid3 = fopen(strcat('output_log_', saveDirName, '.csv'), 'w') ;
fid4 = fopen(strcat('output_LPC_', saveDirName, '.csv'), 'w') ;
fid5 = fopen(strcat('output_area_per_cell_', saveDirName, '.csv'), 'w');
fid6 = fopen(strcat('output_props_', saveDirName, '.csv'), 'w');
fid7 = fopen(strcat('output_area_per_cell_non-ensheathed', saveDirName, '.csv'), 'w');


if length(batch_numFiles) < 2
    for idx = 1:length(all_individual_trials_sheaths)
        
        if isempty(all_individual_trials_sheaths{1, idx})
            all_individual_trials_sheaths{1, idx} = 0;
        end
        
        if isempty(all_individual_trials_lengths{1, idx})
            all_individual_trials_lengths{1, idx} = 0;
        end
        
        if isempty(all_individual_trials_log{1, idx})
            all_individual_trials_log{1, idx} = 0;
        end
        
        if isempty(all_individual_trials_LPC{1, idx})
            all_individual_trials_LPC{1, idx} = 0;
        end
        
        if isempty(all_individual_trials_area_per_cell{1, idx})
            all_individual_trials_area_per_cell{1, idx} = 0;
        end
        
        if isempty(all_individual_trials)
            all_individual_trials = 0;
        end
        
        if isempty(all_individual_trials_area_per_cell{1, idx})
            all_individual_trials_area_per_cell{1, idx} = 0;
        end
        
        if isempty(all_individual_trials_area_per_cell{1, idx})
            all_individual_trials_area_per_cell_non_ensheathed{1, idx} = 0;
        end
        
        dlmwrite(strcat('output_sheaths_', saveDirName, '.csv'), all_individual_trials_sheaths(1, idx), '-append') ;
        dlmwrite(strcat('output_lengths_', saveDirName, '.csv'), all_individual_trials_lengths(1, idx), '-append') ;
        dlmwrite(strcat('output_log_', saveDirName, '.csv'), all_individual_trials_log(1, idx), '-append') ;
        dlmwrite(strcat('output_LPC_', saveDirName,'.csv'), all_individual_trials_LPC(1, idx), '-append') ;
        dlmwrite(strcat('output_area_per_cell_', saveDirName, '.csv'), all_individual_trials_area_per_cell(1, idx), '-append')
        dlmwrite(strcat('output_area_per_cell_non_ensheathed', saveDirName, '.csv'), all_individual_trials_area_per_cell_non_ensheathed(1, idx), '-append')
        dlmwrite(strcat('output_props_', saveDirName, '.csv'), all_individual_trials(idx, :), '-append')
    end
    
else  % if BATCHED with user input
    total_counter = 0;
    for idx = 1:length(batch_numFiles)
        %cycle_files = 0;
        %while cycle_files < batch_numFiles(idx)
        total_counter = total_counter + batch_numFiles(idx);
        
        if isempty(all_individual_trials_sheaths{1, total_counter})
            all_individual_trials_sheaths{1, total_counter} = 0;
        end
        
        if isempty(all_individual_trials_lengths{1, total_counter})
            all_individual_trials_lengths{1, total_counter} = 0;
        end
        
        if isempty(all_individual_trials_log{1, total_counter})
            all_individual_trials_log{1, total_counter} = 0;
        end
        
        if isempty(all_individual_trials_LPC{1, total_counter})
            all_individual_trials_LPC{1, total_counter} = 0;
        end
        
        if isempty(all_individual_trials_area_per_cell{1, total_counter})
            all_individual_trials_area_per_cell{1, total_counter} = 0;
        end
        
        if isempty(all_individual_trials)
            all_individual_trials = 0;
        end
        
        if isempty(all_individual_trials_area_per_cell_non_ensheathed{1, total_counter})
            all_individual_trials_area_per_cell_non_ensheathed{1, total_counter} = 0;
        end
        
        dlmwrite(strcat('output_sheaths_', saveDirName, '.csv'), all_individual_trials_sheaths(1, total_counter), '-append') ;
        dlmwrite(strcat('output_lengths_', saveDirName, '.csv'), all_individual_trials_lengths(1, total_counter), '-append') ;
        dlmwrite(strcat('output_log_', saveDirName, '.csv'), all_individual_trials_log(1, total_counter), '-append') ;
        dlmwrite(strcat('output_LPC_', saveDirName, '.csv'), all_individual_trials_LPC(1, total_counter), '-append') ;
        dlmwrite(strcat('output_area_per_cell_', saveDirName, '.csv'), all_individual_trials_area_per_cell(1, total_counter), '-append')
        dlmwrite(strcat('output_area_per_cell_non_ensheathed', saveDirName, '.csv'), all_individual_trials_area_per_cell_non_ensheathed(1, total_counter), '-append')
        dlmwrite(strcat('output_props_', saveDirName, '.csv'), all_individual_trials, '-append')
        %end
        %cycle_files = cycle_files + 1;
    end
end

fclose(fid1);
fclose(fid2);
fclose(fid3);
fclose(fid4);
fclose(fid5);
fclose(fid6);
fclose(fid7);
fclose(fid_internodes);

%% Make csv files for data analysis
name_csv = 'Result_names.csv';
Row_Names = batch;
T1 = table(Row_Names);

writetable(T1, name_csv, 'WriteRowNames',true);

dlmwrite(name_csv, ' ', '-append');

