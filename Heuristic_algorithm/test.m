%% Need to add:
% 1) Save the "load_five" params ect...
% 2) Get rid of the batching
% 3) %%***NOTE: sheaths have to be in RED channel
% 4) Get areas per cell
% 5) Put in lower threshold for STEM/O4 identification
% 6) Make the near line join thresh smaller
% 7) Get correct scale ==> ***NOTE: changing scale right now changes a LOT
% of stuff... maybe b/c of elim_O4??? or other elim stuff???


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
saveDirName = uigetdir();   % creates directory

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
        sensitivity = calib(diameterFiber, minLength, name, fillHoles, DAPIsize, calibrate, mag, DAPImetric, scale, sensitivity, sigma, foldername, cur_dir);
        defaultans{10} = num2str(sensitivity);
    end
end
cd(saveDirName);
save('___Parameters used___', 'save_params');
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
%hor_factor = 2;
near_join = round(3 / (scale));  % in um
fillHoles = round(8 / (scale * scale));  % in um^2
squareDist = round(50 / (scale));  % in um (is the height of the cell that must be obtained to be considered possible candidate)
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
    
    batch_numFiles = [batch_numFiles, numfids];
    %% Read in images
    for fileNum = 1 : load_five: numfids
        
        cd(cur_dir);
        natfnames=natsort(trialNames);
        %% Decide if want to load individual channels or single image
        if load_five == 5
            [DAPIimage,redImage,binImage,greenImage,wholeImage] = NewFileReaderV4(trialNames, fileNum, allChoices, foldername, cur_dir);  
        else
            % (3) DAPI
            cd(foldername);
            % (4) Red
            filename = natfnames{fileNum};
            redImage = imread(filename);
            %O4_im = im2double(rgb2gray(redImage));
            
        end
        size_red = size(redImage);
        %redImage = redImage(1:square_cut_h, 1: square_cut_w, :);
        square_cut_h = size_red(1);
        square_cut_w = size_red(2);

        width = square_cut_w;
        height = square_cut_h;
        
        %% Store the size of images for batch processing - Tiger 12/02/19
        if fileNum == 1 || length(batch_numFiles) == 1
           batch_sizes = [batch_sizes; [height, width]]; 
        end
        
        cd(cur_dir);
        
        fileNum
        
    end    
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
all_individual_trials_area_per_cell = cell(0);

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
    num_O4_individual = 0;
    num_sheaths_individual = 0;
    
    filename = trialNames{fileNum};
    s = load(filename);
    s = s.allS;
   
    %% use diff sizes taken from the batch-runs if batched - Tiger 13/02/19
    if batch_counter == fileNum - 1 && batch_counter < numfids
        height = batch_sizes(size_counter, 1); % picks new size
        width = batch_sizes(size_counter, 2);
        batch_counter = batch_counter + batch_numFiles(size_counter);
        %batch_counter = batch_counter + 1;
        size_counter = size_counter + 1;
        figure; imshow(tmp);
    end
    
    % Count allNumSheaths and allLengths  ***using LENGTH OF SKELETON
    if ~isempty(s)
        all_area_per_cell = s(1).AreaOverall;
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
                    if len > minLength
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
                    if len > minLength
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
        
    else
        all_individual_trials = [all_individual_trials; [0,  num_O4_individual,num_sheaths_individual]];
    
    end
    num_sheaths_individual
    all_individual_trials_sheaths{end + 1} = allNumSheathsR;
    all_individual_trials_lengths{end + 1} = allLengthFibersR;
    all_individual_trials_log{end + 1} =  allLog;
    all_individual_trials_LPC{end + 1} = allMeanLPC;
    all_individual_trials_area_per_cell{end + 1} = all_area_per_cell;
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
%csvwrite('output_props.csv', all_individual_trials);

fid1 = fopen('output_sheaths.csv', 'w') ;
fid2 = fopen('output_lengths.csv', 'w') ;
fid3 = fopen('output_log.csv', 'w') ;
fid4 = fopen('output_LPC.csv', 'w') ;
fid5 = fopen('output_area_per_cell.csv', 'w');
fid6 = fopen('output_props.csv', 'w');

if length(batch_numFiles) == 1
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
        
        if isempty(all_individual_trials{1, idx})
            all_individual_trials{1, idx} = 0;
        end
        
        dlmwrite('output_sheaths.csv', all_individual_trials_sheaths(1, idx), '-append') ;
        dlmwrite('output_lengths.csv', all_individual_trials_lengths(1, idx), '-append') ;
        dlmwrite('output_log.csv', all_individual_trials_log(1, idx), '-append') ;
        dlmwrite('output_LPC.csv', all_individual_trials_LPC(1, idx), '-append') ;
        dlmwrite('output_area_per_cell.csv', all_individual_trials_area_per_cell(1, idx), '-append') 
        dlmwrite('output_props.csv', all_individual_trials(1, idx), '-append')
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
        
        if isempty(all_individual_trials{1, total_counter})
            all_individual_trials{1, total_counter} = 0;
        end
        
        dlmwrite('output_sheaths.csv', all_individual_trials_sheaths(1, total_counter), '-append') ;
        dlmwrite('output_lengths.csv', all_individual_trials_lengths(1, total_counter), '-append') ;
        dlmwrite('output_log.csv', all_individual_trials_log(1, total_counter), '-append') ;
        dlmwrite('output_LPC.csv', all_individual_trials_LPC(1, total_counter), '-append') ;
        dlmwrite('output_area_per_cell.csv', all_individual_trials_area_per_cell(1, total_counter), '-append')
        dlmwrite('output_props.csv', all_individual_trials(1, total_counter), '-append')
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

%% Make csv files for data analysis
name_csv = 'Result_names.csv';
Row_Names = batch;
T1 = table(Row_Names);

writetable(T1, name_csv, 'WriteRowNames',true);

dlmwrite(name_csv, ' ', '-append');
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

