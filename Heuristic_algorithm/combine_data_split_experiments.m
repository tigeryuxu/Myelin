
original = pwd;

% Go to directory
folder = uigetdir();
cd(folder);

% Get a list of all files and folders in this folder.
files = dir('./');
% Get a logical vector that tells which is a directory.
dirFlags = [files.isdir];
% Extract only those that are directories.
subFolders = files(dirFlags);
% Print folder names to command window.

cur_d = pwd;
cd(cur_d);

% all_folders = cell(0);
% another_f = 'Y';
% while another_f == 'Y'
%     foldername = uigetdir();   % get directory
%     all_folders{end + 1} = foldername;
%     
%     default = {'', 'N'};
%     prompt = {'Another folder?'};
%     dlg_title = 'Input';
%     num_lines = 1;
%     answ = inputdlg(prompt,dlg_title,num_lines,default);
%     another_f = cell2mat(answ(1));
%     cd(foldername);
%     cd('../');
% end
% cd(cur_d);
% 
% subfolders = all_folders;

numTrials = length(subFolders);

%allTrialNames = {'B', 'C', 'PF', 'T', 'Igf'};
%allTrialNames = {'B', 'C', 'O', 'R', 'PF', 'T'};

allTrialNames = {'B', 'C', 'PF', 'T', 'Igf', '211', '222', '255', '311', '322', '355'};


%% REMEMBER FOR BCOR:
% P is C for August 2nd
% T is B (combine)

comb_Lengths = cell(1, length(allTrialNames));
comb_Sheaths = cell(1, length(allTrialNames));
comb_FLC = cell(1, length(allTrialNames));

allFound = cell(0);

for fileNum = 1 : numTrials - 4  % b/c last one is "tmp"
    
    if  isequal(subFolders(fileNum).name, '.') ||  isequal(subFolders(fileNum).name, '..')
        continue;
    end
    
    sub_foldername = strcat('./', subFolders(fileNum).name);
%     sub_foldername = all_folders{fileNum};
    cd(sub_foldername);
    load('Result data.mat');
    
    for i = 1:length(allNames)
        
        char_check = lower(allNames{i});
        char_check(regexp(char_check,'[-]'))=[];  % removes all '-'
        
        char_check = strtrim(char_check); % removes white spaces from ends
        if isequal(char_check, 'ig')
            char_check = 'igf';
        elseif isequal(char_check, 'p')
            char_check = 'pf';
        end
        
        same = strfind(lower(allTrialNames), char_check);   % SHOULD BE ABLE TO FIND "P" in "PF" so they're the same
        idx = find(~cellfun(@isempty,same));
        if idx > 0   % Found something
            
            if idx == 4 || idx == 5
                idx = 1;
            end
            
            allFound{fileNum, idx} = allTrialNames{idx};
         
            %% Combine lengths:
            comb_Lengths{fileNum, idx} =  [comb_Lengths{fileNum, idx}, allTrialLengths{i}];
            
            %% Combine numSheaths
            comb_Sheaths{fileNum, idx} =  [comb_Sheaths{fileNum, idx}, allTrialSheathsR{i}];
            
            %% Combine FLC
            comb_FLC{fileNum, idx} = [comb_FLC{fileNum, idx}, allTrialMeanFLC{i}];
            
        elseif length(idx) > 2
            disp('ERROR');
            disp(idx);
        end
        
        cd(cur_d);
    end
end

mkdir('Combined Results');
cd('./Combined Results');


%% Make csv files for data analysis

% (1) Num_Sheaths
name_csv = 'Result_COMB_num_sheaths.csv';

fileID = fopen(name_csv,'w');    
%dlmwrite(name_csv, 'NumWrapped,', '-append');
for i = 1:length(allTrialNames)
      
%     dlmwrite(name_csv, allTrialNames(i), '-append');
    for j = 1:length(comb_Sheaths(:, i))
    
        name_trial = strcat(allTrialNames{i}, '_', num2str(j));
        fprintf(fileID, '%s,' ,name_trial);
    
        trial_data = cell2mat(comb_Sheaths(j, i));
        for x = 1:length(trial_data)
            fprintf(fileID, '%.2f,', trial_data(x));
            %dlmwrite(name_csv, trial_data,',', j, 2);
        end
        fprintf(fileID, '\n');
        
    end
end
fclose(fileID);

% (2) Lengths
name_csv = 'Result_COMB_lengths.csv';
fileID = fopen(name_csv,'w');    
%dlmwrite(name_csv, 'NumWrapped,', '-append');
for i = 1:length(allTrialNames)
      
%     dlmwrite(name_csv, allTrialNames(i), '-append');
    for j = 1:length(comb_Lengths(:, i))
    
        name_trial = strcat(allTrialNames{i}, '_', num2str(j));
        fprintf(fileID, '%s,' ,name_trial);
    
        trial_data = cell2mat(comb_Lengths(j, i));
        for x = 1:length(trial_data)
            fprintf(fileID, '%.2f,', trial_data(x));
            %dlmwrite(name_csv, trial_data,',', j, 2);
        end
        fprintf(fileID, '\n');
        
    end
end
fclose(fileID);


% (3) mean_FLC
name_csv = 'Result_COMB_mean_FLC.csv';
fileID = fopen(name_csv,'w');    
%dlmwrite(name_csv, 'NumWrapped,', '-append');
for i = 1:length(allTrialNames)
      
%     dlmwrite(name_csv, allTrialNames(i), '-append');
    for j = 1:length(comb_FLC(:, i))
    
        name_trial = strcat(allTrialNames{i}, '_', num2str(j));
        fprintf(fileID, '%s,' ,name_trial);
    
        trial_data = cell2mat(comb_FLC(j, i));
        for x = 1:length(trial_data)
            fprintf(fileID, '%.2f,', trial_data(x));
            %dlmwrite(name_csv, trial_data,',', j, 2);
        end
        fprintf(fileID, '\n');
        
    end
end
fclose(fileID);


cd(original);


