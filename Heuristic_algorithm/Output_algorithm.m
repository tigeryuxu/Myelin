% s = load('1.mat');
% s = s.allS;
% allNumSheathsR = [];
% allLengthFibersR = [];
% allMeanLPC = [];
% allLog = [];
% scale = 0.454;
% 
% % Count allNumSheaths and allLengths  ***using LENGTH OF SKELETON
% for N = 1:length({s.objDAPI})
%     if s(N).Bool_W == 1
%         % Put the fibers into a tmp array so can find MajorAxisLength
%         fibers_cell = [];
%         tmp = zeros([2052, 2052]);
%         for Y = 1:length(s(N).Fibers)
%             tmp(s(N).Fibers{Y}) = 1;
%         end
%         
%         % tmp = imdilate(tmp, strel('disk', 10));
%         [B,L] = bwboundaries(tmp, 'noholes');
%         vv = regionprops(L, 'MajorAxisLength');
%         num_sheaths = 0;
%         for Y = 1:length(vv)
%             len = vv(Y).MajorAxisLength;
%             if len > 25
%                 allLengthFibersR = [allLengthFibersR len * scale] ;
%                 fibers_cell = [fibers_cell len * scale];
%                 log_length = log10(len * scale);
%                 allLog = [allLog log_length];
%                 num_sheaths = num_sheaths + 1;
%             end
%         end
%         
%         if num_sheaths == 0
%             continue
%         end
%         allNumSheathsR = [allNumSheathsR num_sheaths];
%         
%         
%         avg_length = mean(fibers_cell);
%         %s(N).Mean_Fiber_L_per_C = avg_length;  % add to struct
%         
%         allMeanLPC = [allMeanLPC avg_length];
%         
%     end
% end
% 
% 
% 

%% FOR MULTI_FILE COMAPRISON

cur_dir = pwd;
foldername = uigetdir();   % get directory

cd(foldername)
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
cd(foldername);
numfids = length(trialNames);   %%% divided by 5 b/c 5 files per pack currently

36, 44, 91, 93
% allNumSheathsR = [];
% allLengthFibersR = [];
% allMeanLPC = [];
% allLog = [];

width = 6700;   % both should be 2052  or 1600
height = 7200;  % 7729 x 5558

%7345 x 6898

%scale = 0.6904;
scale = 0.454;
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
cd(foldername);
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







