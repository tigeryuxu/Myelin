%% RUN_manual_correct
%Read in images 1 by 1, and then sort through "s" to find the idx that match the image
% Then "crop" up to that "s" to be the "s"
%***should be faster, b/c don't need to run full analysis

scale = 0.227;

cur_d = pwd;
cd(cur_d);

foldername_results = uigetdir();   % choose directiory with "Result data.mat" file
cd(foldername_results);
load('Result data.mat');
cd(cur_d);

foldername = uigetdir(); % choose directory of images
allChoices = choosedialog2();   %% read in choices

saveDirName = create_dir(foldername_results, 'Corrected');
cd(foldername_results);
mkdir(saveDirName);

saveDirName = strcat(foldername_results, '\', saveDirName); 
cd(cur_d);

%% Select which trials to do calibration on:

%name = allNames{i};
%s = allTrialS{i};

defaultans = {' '};

str = {'Current trials are: '};
for i = 1:length(allNames)
    str = strcat(str, allNames{i}, ', ');
end
str = strcat(str, 'Which correct?');

prompt = str;
dlg_title = 'Input';
num_lines = 1;
answer = inputdlg(prompt,dlg_title,num_lines,defaultans);
chr = char(answer); % convert to char array

if findstr(chr, 'all') % Manual correct ALL trials  %%%%%%%%% POSSIBLE FUTURE BUG, don't use findstr, match by deliminated tokens
    in_names = allNames;
else

    in_names = strsplit(chr, ','); % split by commas
end

% Match the names to the trials that want to be corrected
for Y = 1:length(allNames)
    match = 0;
    for i = 1:length(in_names)
        if findstr(allNames{Y}, in_names{i}) % if the same
            match = 1;
        end
    end
    if match == 0 % delete from allNames
        allNames{Y} = [];
        allTrialS{Y} = [];
    end
end
allNames =  allNames(~cellfun('isempty',allNames));   % delete from the list if not a line
allTrialS =  allTrialS(~cellfun('isempty',allTrialS));   % delete from the list if not a line

%% SHUFFLE TO BLIND:

%***of course need at LEAST 2 trials to have a shuffled order
shuffle_order = randperm(length(allNames));
allNames = allNames(shuffle_order);
allTrialS = allTrialS(shuffle_order);

%% Variables
allWrappedR = []; allWrappedG = []; allTotalCells = [];
%allTrialLengths = cell(0);  allTrialSheathsR = cell(0); allTrialSheathsG = cell(0); allInfoInfo = cell(0);
allTrialS_new = cell(0);
%allTrialMeanFLC = cell(0);
allSumO4 = []; allSumMBP = [];

allDiffWR = []; allDiffO4 = [];

term = 0;


%% 
for N = 1:length(allNames)
    
    % Variables
    allInfo = [];
    [sumWrappedR, sumUnWrappedR, sumFibersPerPatchR, sumO4] = deal(0);   % variable declaration
    [sumWrappedG, sumUnWrappedG, sumFibersPerPatchG, sumMBP] = deal(0);
    %allLengthFibers = [];    allLengthFibersG = [];    allNumSheathsR = [];    allNumSheathsG = []; allMeanLPC = [];
    allS_new = [];
    diffWR = 0;
    diffO4 = 0;
    
    % Current trial info
    name = allNames{N};
    s = allTrialS{N};
    
    %% Reads in all the files to analyze in the current directory
    cd(foldername);
    nameCat = strcat(name, '*tif');
    fnames = dir(nameCat);
    
    namecell=cell(1);
    idx = 1;
    for i=1:length(fnames)
        if  isstrprop(fnames(i).name(length(name) + 1), 'digit')
            namecell{idx,1}=fnames(i).name;
            idx = idx + 1;
        end
    end
    trialNames = namecell;
    numfids = length(trialNames);   %%% divided by 5 b/c 5 files per pack currently
    
    for fileNum = 1 : 5 : numfids
        %% Read in images   %%Test 2 is == 4 0 2 1 3  %% Test 1 is == 4 0 3 1 2 %% Test 3 Bright == 0 2 3 4 1   % DIC: 0 1 2 4 3
        cd(cur_d);
        [DAPIimage,redImage,binImage,greenImage,wholeImage] = NewFileReaderV4(trialNames, fileNum, allChoices, foldername, cur_d);
        cd(cur_d);
        
        % whole image
        intensityValueWhole = im2double(rgb2gray(wholeImage));
        lengthY = length(intensityValueWhole(:, 1));
        lengthX = length(intensityValueWhole(1, :));
        
        %% What about the name of the files??? ==> has to be IDENTICAL, to the way the experiments were run, or else "s" is going to be off
        %% ^^^shouldn't be problem, b/c looping through in order of "allNames"
        
        %% FIND matching "s" corresponding to "fileNum"
        im_num = (fileNum + 5 - 1) / 5;
        
        idx_match = find([s.im_num] == im_num);
        
        corr_s = s(idx_match(1) : idx_match(end));
        

       %% ADD NEW FIELD FOR FIBERS:


       [corr_s(:).new_fibers] = deal(0);   % add new field
       
       corr_s(1).new_fibers = cell(0); % for fiber positions
       corr_s(2).new_fibers = cell(0); % for length
    
       %% ENHANCE IMAGE
       r = rgb2gray(redImage);
       d = rgb2gray(DAPIimage);
       
       r = imadjust(r);
       d = imadjust(d);
       
       f = rgb2gray(binImage);
       
       fus = cat(3, r, f, d);

        [corr_s, term] = manual_correct(fus, lengthY, lengthX, scale, corr_s);
        
        % Print results
        %p = strcat(foldername_results, '\', saveDirName);
        cd(saveDirName);
        figure(20);
        filename = strcat('Result', name, num2str((fileNum + 5 - 1)/5)) ;
        print(filename,'-dpng')
        hold off;
        
        %allInfo = [allInfo ; [wrappedR, numCells]];
        allS_new = [allS_new; corr_s];
        
        cd(cur_dir);
        
        % Clears all figures
        arrayfun(@cla,findall(0,'type','axes'));
        
        % Ends program
        if term == 1
            break;
        end
        
    end
    
    %% If terminated early, just run through the rest of the info:
    if term == 1
        
        allS_old = allTrialS{N};
        
        % Find and add the rest of the images (s)
        not_added = allS_old(length(allS_new) + 1 : end);
        
        allS_new = [allS_new; not_added];
    end
    
    % Rest of num wrapped
    idx_wrap = find([allS_new.Bool_W] == 1);
    numWR = length(idx_wrap); % set numWRAPPED
    
    % Rest of DAPI
    numCells = length({allS_new.objDAPI});   % Don't comment out
    
    % Rest of O4+ cells
    numO4 = 0;
    for T = 1:length({allS_new.objDAPI})
        if ~isempty(allS_new(T).Core)
            numO4 = numO4 + 1;
        end
    end
    
    % Difference WR between original and corrected
    idx_wrap = find([s.Bool_W] == 1);
    oldWR = length(idx_wrap); % set numWRAPPED
    diffWR = numWR - oldWR;
    
    % Difference O4+ between original and corrected
    oldO4 = 0;
    for T = 1:length({s.objDAPI})
        if ~isempty(s(T).Core)
            oldO4 = oldO4 + 1;
        end
    end
    diffO4 = numO4 - oldO4;
    
    % Difference total num DAPI???
    
    %% For stats later:
    allWrappedR = [allWrappedR numWR];
    %     allWrappedG = [allWrappedG sumWrappedG];
    allTotalCells = [allTotalCells numCells];
    %     allTrialLengths{i} = allLengthFibers * scale;   %% SCALED values are saved
    %     allTrialSheathsR{i} = allNumSheathsR;
    %     allTrialSheathsG{i} = allNumSheathsG;
    %     allTrialMeanFLC{i} = allMeanLPC;
    allTrialS_new{i} = allS_new;
    
    allSumO4 = [allSumO4 numO4];
    
    allDiffWR = [allDiffWR diffWR];
    allDiffO4 = [allDiffO4 diffO4];
    %     allSumMBP = [allSumMBP sumMBP];
    
    %     allInfoInfo{i} = allInfo;
    
    cd(saveDirName);
    
    %   (2) "Summary.txt" is for summary of ALL the images
    proportionR = numWR/numCells;
    %     proportionG = sumWrappedG/(sumUnWrappedG + sumWrappedG);
    
    nameTmp = strcat('summary', name, '.txt');
    fileID = fopen(nameTmp,'w');
    fprintf(fileID,'Total num images analyzed: %d \n', numfids/5);
    fprintf(fileID,'Num wrapped R cells: %d \n', numWR);
    %     fprintf(fileID,'Num un-wrapped R cells: %d \n', sumUnWrappedR);
    fprintf(fileID,'Proportion wrapped R: %.2f \n', proportionR);
    
    %     fprintf(fileID,'Num wrapped G cells: %d \n', sumWrappedG);
    %     fprintf(fileID,'Num un-wrapped G cells: %d \n', sumUnWrappedG);
    %     fprintf(fileID,'Proportion wrapped G: %.2f \n\n', proportionG);
    
    fprintf(fileID,'Proportion Wrapped / O4+ cells: %.2f \n', numWR/ numO4);
    fprintf(fileID,'Proportion O4+ / Total cells: %.2f \n', numO4 / numCells);
    fprintf(fileID,'Total num O4+ cells: %d \n', numO4);
    fprintf(fileID,'Total num cells: %d \n', numCells);
    
    fprintf(fileID,'Difference corrected O4+: %.2f \n', diffO4);
    fprintf(fileID,'Difference corrected WR: %.2f \n', diffWR);
    
    %     fprintf(fileID,'Total num R Fibers: %d \n', sumFibersPerPatchR);
    %     fprintf(fileID,'Avg length of wrapping per R fiber: %f \n\n', mean2(allLengthFibers));
    
    %     fprintf(fileID,'Total num G Fibers: %d \n', sumFibersPerPatchG);
    %     fprintf(fileID,'Avg length of wrapping per G fiber: %f \n', mean2(allLengthFibersG));
    
    %     fprintf(fileID,'Sensitivity of line segmentation: %.2f \n', sensitivity);
    
    %     fprintf(fileID,'User selected parameters %s \n', cell2mat(defaultans));
    
    fclose(fileID);
    
    if term == 1
        break;
    end
    
end

%% Save results
clearvars -except allNames allTrialLengths allTrialSheathsR allTrialSheathsG allWrappedR allWrappedG allTotalCells...
    allInfoInfo allSumO4 allSumMBP saveDirName cur_dir norm_props_R norm_props_G squareDist allTrialS allTrialMeanFLC...
    allDiffWR allDiffO4

cd(saveDirName);

%% Make Table
allWrappedR = allWrappedR';
allSumO4 = allSumO4';
allTotalCells = allTotalCells';
allWrappedG = allWrappedG';
allNames = allNames';
allDiffWR = allDiffWR';
allDiffO4 = allDiffO4';
% norm_props_R = norm_props_R';
% norm_props_G = norm_props_G';

propW_O4 = allWrappedR./allSumO4 * 100;

propW_total = allWrappedR./allTotalCells * 100;
save('Result data');

%T1 = table(norm_props_R, norm_props_G, propW_O4, allWrappedR, allSumO4, allTotalCells, propW_total, allWrappedG,'RowNames', allNames)
T1 = table(propW_O4, allWrappedR, allSumO4, allTotalCells, allDiffWR, allDiffO4, 'RowNames', allNames)

writetable(T1, 'Result_table.csv', 'WriteRowNames',true);

%% Make csv files for data analysis
name_csv = 'Result_names.csv';
Row_Names = allNames;
T1 = table(Row_Names);

writetable(T1, name_csv, 'WriteRowNames',true);

dlmwrite(name_csv, ' ', '-append');

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
cd(cur_dir);





