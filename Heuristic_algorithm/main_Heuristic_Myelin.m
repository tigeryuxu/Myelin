%% Need to add:
% 1) Save the "load_five" params ect...
% 2) Get rid of the batching

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
    
    nanoYN = get((output.checkbox12), 'value');
    combineRG = get((output.Combine_RG), 'value');
    verbose = get((output.verbose), 'value');
    calibrate = get((output.calib), 'value');
    match_words = get((output.match_full_name), 'value');
    bool_load_five = get((output.checkbox11), 'value');
    adapt_his = get((output.Nano_YN), 'value');
    divide_im = get((output.checkbox13), 'value');
    
    save_params = {name, batch, scale, diameterFiber, sigma, sensitivity, minLength, DAPIsize, nanoYN...
        ,verbose, calibrate, match_words, bool_load_five, adapt_his, divide_im};
    
    if calibrate
        sensitivity = calib(diameterFiber, minLength, name, fillHoles, DAPIsize, calibrate, mag, DAPImetric, scale, sensitivity, sigma, foldername, cur_dir);
        defaultans{10} = num2str(sensitivity);
    end
end
cd(saveDirName);
save('Parameters used', 'save_params');
cd(cur_dir);

%% TO ADD as GUI prompts:
%square_cut_h = 6700; % 2052 for 8028 images, and %1600 for newer QL images  and 1500 for HA751
%square_cut_w = 7200;

square_cut_h = 1460; % 2052 for 8028 images, and %1600 for newer QL images  and 1500 for HA751
square_cut_w = 1936;

enhance_RED = 'N'; % ==> set as 'Y' for other human OL trials!!!
remove_nets = 'Y'; % background subtraction for O4

mag = 'N';
enhance = 'Y';   % (Background subtraction for DAPI + O4 images) CLEM ==> doesn't need this

%% PRESET and scaled
DAPImetric = 0.3;   % Lower b/c some R not picked up due to shape...
percentDilate = 2;   % for cores
hor_factor = 2;
near_join = round(10 / (scale));  % in um
fillHoles = round(8 / (scale * scale));  % in um^2
squareDist = round(50 / (scale));  % in um (is the height of the cell that must be obtained to be considered possible candidate)
coreMin = 0;
elim_O4 = round(20 / (scale * scale));    % SMALL O4 body size (200 ==> 1000 pixels)

%% ADD TO MENU:
if bool_load_five == 1
    load_five = 5;
else
    load_five = 1;
end

if load_five == 5
    allChoices = choosedialog2();   %% read in choices  %%% SWITCH TO FROM GUI.m
end

batch_skip = 'N';
batch_run = 'N';
batch_num = 0;
batch = cell(2);   % intialize empty

% Clem:
%batch = {'Clem1-', 'Clem2-', 'Ctr'};
%save_params = {'', '', '0.454', '10', '4', '0.9', '35', '25', '0', '0', '0', '0', '0'};
%save_params = {'', '', '0.227', '20', '8', '0.9', '12', '8', '0', '0', '0', '0', '0'};

%%
%batch = {'*Olig2_WT', '*Olig2_KO'};
%batch = {'*KOSkap2_20x', '*WT_20x'};
%batch = {''};

%batch = {'*C1', '*C2', '*C3', '*RR1', '*RR2', '*RR31'};

%batch = {'n1_KO', 'n1_WT', 'n2_KO', 'n2_WT', 'n3_20xzoom_MBP_KO',  'n3_20xzoom_MBP_WT', 'n4_20x_zoom_KO', 'n4_20x_zoom_WT'};

batch = {'n1_20x_KO', 'n1_20x_WT', 'n2_KOSkap2_20x', 'n2_WT_20x', 'n3_20x_snap_MBP_CD140_WT_', 'n3_20x_snap_MBP_CD140_KO_',  'n3_snap_20x_MBP_Olig2_KO_', 'n3_snap_20x_MBP_Olig2_WT_',   'n4_20x_MBP_KO', 'n4_20x_MBP_WT', 'n5_KO', 'n5_WT'};


%% Run Analysis
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
    nameTmp = strcat('allAnalysis', name, '.txt');   % Output file
    fileID = fopen(nameTmp,'w');
    cd(cur_dir);
    
    cd(foldername);   % switch directories
    nameCat = strcat(name, '*tif');
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
    
    %% Read in images
    for fileNum = 1 : load_five: numfids
        
        cd(cur_dir);
        natfnames=natsort(trialNames);
        %% Decide if want to load individual channels or single image
        if load_five == 5
            [DAPIimage,redImage,binImage,greenImage,wholeImage] = NewFileReaderV4(trialNames, fileNum, allChoices, foldername, cur_dir);
            DAPIimage = rgb2gray(DAPIimage);
            red = rgb2gray(redImage);
            binImage = rgb2gray(binImage);
            greenImage = rgb2gray(greenImage);
            
            cd(cur_dir);
            
            %(1) Fibers
            nanoF_im = im2double(binImage);     % Reads-in Bin
            siz = size(nanoF_im);
            
            % decides whether or not to include nanoF_im
            if nanoYN == 1
                nanoF_im = adapthisteq(nanoF_im);   % adjust
                if mag == 'Y'
                    se = strel('disk', 30);    % maybe make the same size as the fibers???
                else
                    se = strel('disk', 8);
                end
                I = nanoF_im;
                J = imsubtract(imadd(I,imtophat(I,se)),imbothat(I,se));    % top-hat AND bottom-hat
                J = imerode(J, strel('disk', 6));
                nanoF_im = J;
                phaseThres = graythresh(nanoF_im);
                nanoF_im = imbinarize(nanoF_im);  % binarizes
                figure(2); imshow(nanoF_im);
            else
                nanoF_im = false(siz);
            end
            
            redImage = cat(3, red, greenImage, DAPIimage);
            
            red = redImage(:, :, 1);
            if enhance_RED == 'Y'
                red = imadjust(red);
            else
                red = adapthisteq(red);
            end
            wholeImage = cat(3, red, adapthisteq(greenImage), adapthisteq(DAPIimage));
            
            
        else
            % (3) DAPI
            cd(foldername);
            % (4) Red
            filename = natfnames{fileNum};
            redImage = imread(filename);
            %O4_im = im2double(rgb2gray(redImage));
            
        end
        
        
        % decide if want to run as WHOLE image, or cut into squares
        if divide_im == 0
            size_red = size(redImage);
            %redImage = redImage(1:square_cut_h, 1: square_cut_w, :);
            square_cut_h = size_red(1);
            square_cut_w = size_red(2);
        else
            redImage = redImage(1:square_cut_h * 4, 1:square_cut_w * 4, :);
        end
        width = square_cut_w;
        height = square_cut_h;
        
        
        intensityValueDAPI = im2double(redImage(:,:,3));
        O4_original = im2double(redImage(:,:,1));
        
        
        DAPIimage = redImage(:, :, 3);
        DAPIimage = im2double(DAPIimage);
        %intensityValueDAPI = im2double(rgb2gray(DAPIimage));
        
        % (4) Red
        %filename = natfnames{fileNum + 1};
        redImage = redImage(:, :, 1);
        redImage = im2double(redImage);
        
        
        if enhance_RED == 'Y'
            [all_O4_im_split] = split_imV2(redImage, square_cut_h, square_cut_w);
            
            redImage = imadjust(redImage);
        end
        %O4_im = im2double(rgb2gray(redImage));
        
        
        %% SPLIT IMAGE
        cd(cur_dir);
        [all_DAPI_im_split] = split_imV2(DAPIimage, square_cut_h, square_cut_w);    % into 64 images (2 ^ 3) ==> 1026,    % into 16 images ==> 2052
        
        [all_O4_im_split] = split_imV2(redImage, square_cut_h, square_cut_w);
        
        if load_five == 1
            greenImage = redImage;
        end
        [all_MBP_im_split] = split_imV2(greenImage, square_cut_h, square_cut_w);
        
        %% LOOP
        counter = 0;
        for Q = 1:length(all_DAPI_im_split(:, 1))
            
            for R = 1:length(all_DAPI_im_split(1, :))
                if isempty(all_DAPI_im_split{Q, R})
                    continue;
                end
                
                intensityValueDAPI = im2double(all_DAPI_im_split{Q, R});
                O4_original = im2double(all_O4_im_split{Q, R});
                MBP_im = im2double(all_MBP_im_split{Q, R});
                
                %% IF WANT TO DO WHOLE IMAGE, uncomment these two
                %intensityValueDAPI = DAPIimage;
                %O4_original = redImage;
                
                %% 2018-11-13: MADE THE BACKGROUND SUBTRACTION LARGER so won't lose pieces of cell in analysis
                O4_im_ridges = O4_original;
                [O4_im, originalRed] = imageAdjust(O4_original, fillHoles, enhance);   % image adjust
                
                if enhance_RED == 'Y'
                    O4_im = imclose(O4_im, strel('disk', 10));
                end
                
                %% 2018-12-14: ADDED??? b/c should be looking at ridges from the actual original image??? not binarized???
                O4_im_ridges_adapted = O4_original;
                
                %% 2018-12-14: Can remove the weird membrane between sheaths
                if remove_nets == 'Y'
                    tmp_im = O4_original;
                    background = imopen(O4_original,strel('disk',10));
                    I2 = O4_original - background;
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
                figure(1); imshow(combined_im);
                
                siz = size(combined_im);
                
                %% (1) Find peaks for DAPI
                %DAPIsize = 10;
                [mat, objDAPI, DAPI_bw] = DAPIcount_2(intensityValueDAPI, DAPIsize, DAPImetric, enhance, siz);  % function
                
                if length(objDAPI) > 8000
                    continue;
                end
                
                
                %% Initializes struct to store everything
                c= cell(length(objDAPI), 1); % initializes Bool_W with all zeros
                [c{:}] = deal(0);
                strucMat = num2cell(mat, 2);
                s = struct('objDAPI', objDAPI', 'centerDAPI', strucMat, 'Core', cell(length(objDAPI), 1)...
                    ,'CB', cell(length(objDAPI), 1), 'Fibers', cell(length(objDAPI), 1), 'Mean_Fiber_L_per_C', cell(length(objDAPI), 1), 'Bool_W', c...
                    , 'im_num', c, 'O4_bool', c);
                
                %% (2) Extract cores
                
                [s] = reg_core_filt(combined_im, diameterFiber, siz, percentDilate, s);  % small cores
                [cb, no_dilate_cb, s] = cell_body_filt(combined_im, diameterFiber, siz, coreMin, s);        % cell body
                
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
                [cores, cb, unass_cb, s] = O4_count(combined_im, cores, cb, siz,diameterFiber, elim_O4, s);
                
                if verbose
                    figure(70); imshow(unass_cb); title('Unassociated Objects');
                end
                
                %% Print * for DAPI and O4+
                if enhance_RED == 'Y'
                    tmpDAPI = imadjust(intensityValueDAPI);
                else
                    tmpDAPI = adapthisteq(adapthisteq(adapthisteq(intensityValueDAPI)));
                    O4_original = adapthisteq(adapthisteq(adapthisteq(O4_original)));
                end
                wholeImage = cat(3, O4_original, zeros(size(O4_im)), tmpDAPI);
                
                figure(5); imshow(wholeImage); title('Output Image'); hold on;
                for Y = 1:length({s.objDAPI})
                    if ~isempty(s(Y).centerDAPI) % Print DAPI
                        text(s(Y).centerDAPI(1, 1),  s(Y).centerDAPI(1 ,2), '^',  'color','m' ,'Fontsize',8);   % writes "peak" besides everything
                    end
                    
                    if ~isempty(s(Y).Core)  % Print O4+
                        text(s(Y).centerDAPI(1, 1),  s(Y).centerDAPI(1 ,2), 'O4',  'color','y' ,'Fontsize',5);   % writes "peak" besides everything
                        sumO4 = sumO4 + 1;
                        s(Y).O4_bool = 1;
                    end
                end
                
                %% (4) Sort through and find cell objects that are much too small, and set them permanently to NOT wrapped
                new_combined_im = imbinarize(unass_cb + combined_im);     % new combined_im also must include the dilated image in O4_count
                
                %% 2018-11-13: REMOVED A SETTING FOR HUMAN OLs, now will allow OLs that look like thin lines
                [s] = small_del_O4(new_combined_im, minLength, squareDist, siz, s);
                
                %% (5) Line seg:
                [fibers, fibers_idx, Lambda2R] = line_seg(O4_im_ridges_adapted, zeros(size(O4_im)), sigma, siz, sensitivity);
                %         if nanoYN == 'Y' % only if there are fibers
                %             fibers = imopen(fibers, strel('disk', 3));    % first erodes away some holes
                %             fibers = imclose(fibers, strel('disk', 3));    % then CLOSES HOLES
                %         end
                %
                %         if verbose
                %             figure(24); imshow(fibers)
                %         end
                %
                %% (6) Clean fibers by subtracting out CB
                fibers = imbinarize(fibers - cb);
                if mag == 'Y'
                    fibers = imopen(fibers, strel('disk', 2));   % to get rid of straggling thin strands
                end
                
                %% (7) NEW LINE ANALYSIS (transforms ridges to lines)
                % Horizontal lines are more dim
                [all_lines, locFibers,allLengths, mask, fibers] = ridges2lines(fibers, siz, hor_factor, minLength);
                
                locFibers =  locFibers(~cellfun('isempty',locFibers));   % delete from the list if not a line
                allLengths = allLengths(~cellfun('isempty', allLengths));   % delete from the list if not a line
                
                %% (8) Check CBs to see if wrapped or not
                fibers_sub_cb = bwmorph(cb, 'thicken', 3);
                [locFibers, allLengths, s] = wrappingAnalysis(fibers_sub_cb, locFibers, allLengths, siz, minLength, isGreen, s);
                
                %% (9) Check remaining fibers with fibers_sub_cb (real), to see if wrapped or not
                % COUNT AGAIN, with a FULL fibers_sub_cb, to get all the fibers NOT directly connected to stuff
                % and use only the REMAINING fibers (i.e. fibers(locFibers) = 0) ==> set to zero the already found ones
                fibers_sub_cb = imbinarize(combined_im - cb);  % THE REAL FIBERS_sub_cb
                [sub_locFibers, allLengths, s] = wrappingAnalysis(fibers_sub_cb, locFibers, allLengths, siz, minLength, isGreen, s);
                
                locFibers = sub_locFibers; % THESE ARE THE FIBERS STILL UN-ASSOCIATED
                
                %% (10) Associate the remaining lines with the nearest DAPI point
                [locFibers, s] = near_line_join(locFibers, near_join, siz, verbose, s);
                
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
                
                
                if load_five == 5 && fileNum > 1
                    fileNum_sav = ((fileNum - 1)/load_five) + 1;
                else
                    fileNum_sav = (fileNum);
                end
                
                %% Print images of results
                cd(saveDirName);
                figure(5);
                filename = strcat('Result', name, num2str(fileNum_sav), '_', num2str(counter));
                print(filename,'-dpng')
                hold off;
                
                figure(1);
                filename = strcat('Result', name, num2str(fileNum_sav),  '_', num2str(counter), 'Combined_im') ;
                print(filename,'-dpng')
                hold off;
                
                % Saves image
                figure(100);
                filename = strcat('Result', name, num2str(fileNum_sav),  '_', num2str(counter), 'DAPI') ;
                print(filename,'-dpng')
                hold off;
                
                figure(31);
                filename = strcat('Result', name, num2str(fileNum_sav),  '_', num2str(counter), 'final_fibers') ;
                print(filename,'-dpng')
                hold off;
                
                figure(67); imshowpair(wholeImage, fibers); title('Filter ridges');
                filename = strcat('Result', name, num2str(fileNum_sav),  '_', num2str(counter), 'Filter ridges') ;
                print(filename,'-dpng'); hold off;
                
                figure(88); imshowpair(wholeImage, mask); title('Ridges to lines after sub core');  hold on;
                filename = strcat('Result', name, num2str(fileNum_sav),  '_', num2str(counter), 'Skeletonized ridges') ;
                print(filename,'-dpng'); hold off;
                
                %% Print to file:
                %(1) "allAnalysis.txt" is for EACH image
                
                %%if file is already open, then no need to open it again
                fprintf(fileID,'Image No. : %d              File name: %s \n', (fileNum_sav), name);
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
            end
        end
        
        cd(saveDirName);
        save(num2str(fileNum_sav), 'allS');
        allS = [];
        cd(cur_dir);
        
    end
    fclose(fileID);      %%%close after so it can append, but next time it writes file, it will over-write    %%% in the future maybe just append??? and have hours log
    
    cd(saveDirName);
    %% (2) "Summary.txt" is for summary of ALL the images
    proportionR = sumWrappedR/(sumUnWrappedR + sumWrappedR);
    proportionG = sumWrappedG/(sumUnWrappedG + sumWrappedG);
    
    nameTmp = strcat('summary', name, '.txt');
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

for fileNum = 1 : numfids
    
    allNumSheathsR = [];
    allLengthFibersR = [];
    allMeanLPC = [];
    allLog = [];
    num_O4_individual = 0;
    num_sheaths_individual = 0;
    
    filename = trialNames{fileNum};
    s = load(filename);
    s = s.allS;
    
    % Count allNumSheaths and allLengths  ***using LENGTH OF SKELETON
    if ~isempty(s)
        for N = 1:length({s.objDAPI})
            
            if isfield(s, 'O4_bool') && s(N).O4_bool
                numO4 = numO4 + 1;
                num_O4_individual = num_O4_individual + 1;
            end
            
            if s(N).Bool_W == 1
                % Put the fibers into a tmp array so can find MajorAxisLength
                fibers_cell = [];
                tmp = zeros([width, height]);
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
        all_individual_trials = [all_individual_trials; [length({s.objDAPI}), num_O4_individual,num_sheaths_individual,num_sheaths_individual/ num_O4_individual* 100 ]];
        
    else
        
        all_individual_trials = [all_individual_trials; [0,  num_O4_individual,num_sheaths_individual,num_sheaths_individual/ num_O4_individual* 100 ]];
    end
    num_sheaths_individual
    all_individual_trials_sheaths{end + 1} = allNumSheathsR;
    all_individual_trials_lengths{end + 1} = allLengthFibersR;
    all_individual_trials_log{end + 1} =  allLog;
    all_individual_trials_LPC{end + 1} = allMeanLPC;
    
end

cd(cur_dir);

% allNumSheathsR = allNumSheathsR';
% allLengthFibersR = allLengthFibersR';
% allMeanLPC = allMeanLPC';
% allLog = allLog';
%
% propWrap = length(allNumSheathsR)/numO4
% numO4
% length(allNumSheathsR)

% SAVE CSV FOR ALL INDIVIDUAL TRIALS
cd(saveDirName);
csvwrite('output_props.csv', all_individual_trials);


fid1 = fopen('output_sheaths.csv', 'w') ;
fid2 = fopen('output_lengths.csv', 'w') ;
fid3 = fopen('output_log.csv', 'w') ;
fid4 = fopen('output_LPC.csv', 'w') ;

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
    
    dlmwrite('output_sheaths.csv', all_individual_trials_sheaths(1, idx), '-append') ;
    dlmwrite('output_lengths.csv', all_individual_trials_lengths(1, idx), '-append') ;
    dlmwrite('output_log.csv', all_individual_trials_log(1, idx), '-append') ;
    dlmwrite('output_LPC.csv', all_individual_trials_LPC(1, idx), '-append') ;
    
    
end
fclose(fid1);
fclose(fid2);
fclose(fid3);
fclose(fid4);

%% Prompt to plot everything:
% questTitle='Plot Data?';
% start(timer('StartDelay',1,'TimerFcn',@(o,e)set(findall(0,'Tag',questTitle),'WindowStyle','normal')));
% button3 = questdlg('Plot Data?', questTitle, 'Y','N','Y','Y');
% ynPlot = button3;
%
% ynPlot = 'Y';
%
% if ynPlot == 'Y'
%
%     defaultans = {'', 'N'};
%     prompt = {'Which trial number was control?'};
%     dlg_title = 'Input';
%     num_lines = 1;
%     answer = inputdlg(prompt,dlg_title,num_lines,defaultans);
%     control_idx = str2double(cell2mat(answer(1)));
%
%     %control_idx = 1;
%
%     name = 'O4';
%     norm_props_R = plotProps(allNames, allWrappedR, allSumO4, allTotalCells, name, control_idx, saveDirName);
%     cd(cur_dir);
%
%     name = 'MBP';
%     norm_props_G = plotProps(allNames, allWrappedG, allSumO4, allTotalCells, name, control_idx, saveDirName);
%     cd(cur_dir);
%
%     %% Calls function
%     plotData(allNames, allTrialLengths, allTrialSheathsR, allTrialSheathsG, saveDirName, cur_dir);
%     cd(cur_dir);
%
%     %% Save results
%     clearvars -except allNames allTrialLengths allTrialSheathsR allTrialSheathsG allWrappedR allWrappedG allTotalCells...
%         allInfoInfo allSumO4 allSumMBP saveDirName cur_dir norm_props_R norm_props_G squareDist allTrialS allTrialMeanFLC
%     cd(saveDirName);
%
%
%     %% Make Table
%
%     allWrappedR = allWrappedR';
%     allSumO4 = allSumO4';
%     allTotalCells = allTotalCells';
%     allWrappedG = allWrappedG';
%     allNames = allNames';
%     norm_props_R = norm_props_R';
%     norm_props_G = norm_props_G';
%
%     propW_O4 = allWrappedR./allSumO4 * 100;
%
%     propW_total = allWrappedR./allTotalCells * 100;
%     save('Result data');
%
%     T1 = table(norm_props_R, norm_props_G, propW_O4, allWrappedR, allSumO4, allTotalCells, propW_total, allWrappedG,'RowNames', allNames);
%
%     writetable(T1, 'Result_table.csv', 'WriteRowNames',true);
%
%     %% Make csv files for data analysis
%     name_csv = 'Result_names.csv';
%     Row_Names = allNames;
%     T1 = table(Row_Names);
%
%     writetable(T1, name_csv, 'WriteRowNames',true);
%
%     dlmwrite(name_csv, ' ', '-append');
%
%     name_csv = 'Result_num_sheaths.csv';
%     %dlmwrite(name_csv, 'NumWrapped,', '-append');
%     for i = 1:length(allNames)
%
%         row = i + 1;
%         col = 2;
%
%         trial_data = cell2mat(allTrialSheathsR(i));
%         dlmwrite(name_csv, trial_data, '-append');
%     end
%
%
%     name_csv = 'Result_lengths.csv';
%     for i = 1:length(allNames)
%
%         row = i + 1;
%         col = 2;
%
%         trial_data = cell2mat(allTrialLengths(i));
%         dlmwrite(name_csv, trial_data, '-append');
%     end
%
%     name_csv = 'Result_mean_FLC.csv';
%     for i = 1:length(allNames)
%
%         row = i + 1;
%         col = 2;
%
%         trial_data = cell2mat(allTrialMeanFLC(i));
%         dlmwrite(name_csv, trial_data, '-append');
%     end
%
%     cd(cur_dir);
% end

