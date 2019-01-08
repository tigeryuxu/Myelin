% Read structs:
load('Result data.mat');

trial_sum = cell(0);
for i = 1:length(allTrialS)
    trialS = allTrialS{i};
    totalImages = 0;
    
    allEnsheath = [];
    allO4 = [];
    allDAPI = [];
   
    im_Ensheath = 0;
    im_O4 = 0;
    im_DAPI = 0;
    for Y = 2 : length(trialS)
        check = trialS(Y).im_num;
        if Y + 1 < length(trialS) && check == trialS(Y + 1).im_num   % same numbers:
            totalImages = totalImages + 1;
            
            if trialS(Y).Bool_W == 1
                im_Ensheath = im_Ensheath + 1;
            end
            
            if ~isempty(trialS(Y).Core)
                im_O4 = im_O4 + 1;
            end
            
            im_DAPI = im_DAPI + 1;
            
        else
             if trialS(Y).Bool_W == 1
                im_Ensheath = im_Ensheath + 1;
            end
            
            if ~isempty(trialS(Y).Core)
                im_O4 = im_O4 + 1;
            end
            
            im_DAPI = im_DAPI + 1;
            
            
            allEnsheath = [allEnsheath, im_Ensheath];
            allO4 = [allO4, im_O4];
            allDAPI = [allDAPI, im_DAPI];
            
            im_Ensheath = 0;
            im_O4 = 0;
            im_DAPI = 0;
            
        end
        
    end
    
    allProp = allEnsheath ./ allO4;
    
    
    combine_d = allO4(1, 1:end-1);
    combine_d(2, :) = allEnsheath(1, 1:end-1);
    combine_d(3, :) = allDAPI(1, 1:end-1);
    combine_d(4, :) = allProp(1, 1:end-1);
    
    trial_sum{i} = combine_d';
    
end