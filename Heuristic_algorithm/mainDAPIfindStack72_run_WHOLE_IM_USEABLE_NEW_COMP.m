%(3) z - stack
% (4) ***need stats
% (7) eventually add try/catch

%% Priorities:
% (1) ==> some of the newly counted O4 cores ARE disjoint fibers...
% thus get read falsely as "wrapped"
% ahhhhh, thus may need to do cell_body_filt on new cores...??? 
% OR, associate fibers FIRST, and then run analysis on these cells???

% (2) get rid of single fibers hugging edge of cell near DAPI center
% maybe try subtracting out a rectangle
% OR, try increasing "diameterFiber"

%% FIX + CHANGE
% Need to make ridges2line better
% but, also try to come up with way to melt everything together:
% 
% (1) High magnification ==> oldest images
% - need high density, need...
% (2) Low magnification, high density
% (3) Low magnification, low density
% (4) Clem ==> no enhance DAPI


load_five = true   % DECIDE IF WANT TO LOAD 5 or 1 image


opengl hardware;
close all;

cur_dir = pwd;
cd(cur_dir);

%% For stats
allWrappedR = []; allWrappedG = []; allTotalCells = [];
allNames = cell(0); allTrialLengths = cell(0);  allTrialSheathsR = cell(0); allTrialSheathsG = cell(0); allInfoInfo = cell(0);
allTrialS = cell(0); allTrialMeanFLC = cell(0);
allSumO4 = []; allSumMBP = [];

foldername = uigetdir();   % get directory
allChoices = choosedialog2();   %% read in choices  %%% SWITCH TO FROM GUI.m

moreTrials = 'Y';
trialNum = 1;

saveName = strcat(foldername, '_');
saveDirName = create_dir(cur_dir, saveName);   % creates directory
mkdir(saveDirName);

%% Prompt

%%
% '10', '35', 'N', '25', '25', 'N', 'N', '0.25', '0.227', '0.8', '4', 'N', 'N', '75', 'N'
% '20', '55', 'N', '150', '300', 'N', 'Y', '0.6', '0.227', '0.9', '8', 'N', 'N', '200', 'N'


minLength = 12;  % microns
DAPIsize = 15; % microns ^ 2

%% OLD PARAMETERS:
save_params = {'', '', '0.227', '20', '8', '0.9', '55', '300', '0', '0', '0', '0', '0'};
%save_params = {'', '', '0.178', '18', '10.2', '0.9', '70.14', '382.6', '0', '0', '0', '0', '0'};
save_params = {'', '', '0.178', '18', '8', '1.8', '70.14', '382.6', '0', '0', '0', '0', '0'};
save_params = {'', '', '0.089', '36', '20.4', '0.9', '140.24', '765.2', '0', '0', '0', '0', '0'};
save_params = {'', '', '0.089', '36', '10', '1.8', '140.24', '765.2', '0', '0', '0', '0', '0'};
save_params = {'', '', '0.454', '10', '4', '0.9', '35', '25', '0', '0', '0', '0', '0'};

%% NEW SCALED:
save_params = {'', '', '0.227', '20', '8', '0.9', '12', '15', '0', '0', '0', '0', '0'};
%save_params = {'', '', '0.178', '18', '10.2', '0.9', '70.14', '382.6', '0', '0', '0', '0', '0'};
save_params = {'', '', '0.178', '18', '8', '1.8', '12', '15', '0', '0', '0', '0', '0'};

%save_params = {'', '', '0.089', '36', '20.4', '0.9', '12', '15', '0', '0', '0', '0', '0'};
save_params = {'', '', '0.089', '36', '10', '1.0', '12', '15', '0', '0', '0', '0', '0'};
save_params = {'', '', '0.454', '10', '4', '0.9', '12', '15', '0', '0', '0', '0', '0'};

save_params = {'', '', '0.312', '14', '8', '1.0', '12', '15', '0', '0', '0', '0', '0'};
% MOUSE == 10 squareDist
% RAT == 0.45.4

save_params = {'', '', '0.6904', '10', '6', '0.9', '12', '15', '0', '0', '0', '0', '0'};

% DAPIsize = 10;
% coreMin = 20;
% diameterFiber = 10;
% 
% sigma = 6;
% minLength = 20;
% sensitiviy = 0.9;

%1.8
% gauss = 8
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
    minLength = str2double(get((output.minLength), 'String')) / scale;
    DAPIsize = str2double(get((output.DAPIsize), 'String')) / (scale * scale);
    
    nanoYN = get((output.Nano_YN), 'value');
    combineRG = get((output.Combine_RG), 'value');
    verbose = get((output.verbose), 'value');
    calibrate = get((output.calib), 'value');
    match_words = get((output.match_full_name), 'value');
    
    save_params = {name, batch, scale, diameterFiber, sigma, sensitivity, minLength, DAPIsize, nanoYN...
        ,verbose, calibrate, match_words};
    
    if calibrate
        fillHoles = 0; mag = 0; DAPImetric = 0;
        [sensitivity, sigma, calibrate] = calib(diameterFiber, minLength, name, fillHoles, DAPIsize, calibrate, mag, DAPImetric, scale, sensitivity, sigma, foldername, cur_dir);
        defaultans{10} = num2str(sensitivity);
        defaultans{9} = num2str(sigma);
        
        %% need to add a way to update the GUI with these new calibrated values
    end
end

batch_skip = 'N';
batch_run = 'N';
batch_num = 0;
batch = cell(2);   % intialize empty

%%
%batch = {'*Olig2_WT', '*Olig2_KO'};
batch = {''};

%batch = {'*C1', '*C2', '*C3', '*RR1', '*RR2', '*RR31'};

%% TO ADD:
% Clem is ==> mag (Y), dense (Y), enhance (N), combineRG (Y)
mag = 'Y';
dense = 'Y';             % for different types of images
DAPImetric = 0.3;   % Lower b/c some R not picked up due to shape...
percentDilate = 2;   % for cores
calibrate = 1;
enhance = 'Y';   % CLEM ==> doesn't need this
if mag == 'Y'  && dense == 'Y'            %%%%%%%%%%%%% DENSE
    adapt_his = 0;  % don't enhance fibers
    squareDist = 10 / scale;
    coreMin = 0;         % for cleaning CBs
    
    %% ^^^originall 1000
    %near_join = 200;
    hor_factor = 1;
    %%
    near_join = 11.35 / scale;       %%%%%%%%%%SWITCHED TO 10
    fillHoles = round(7.73 / (scale * scale));
    
elseif mag == 'Y' && dense == 'N'        %%%%%%%%%%%%% NOT DENSE
    % minLength exclusion
    % adapthisteq
    % DAPI_exclusion
    adapt_his = 1;
    squareDist = 200;
    coreMin = 200;         % for cleaning CBs
    near_join = 50;
    fillHoles = 150;
    hor_factor = 2;
    
elseif  mag == 'N' && dense == 'N'           %%%%%%%%%%%% OLD IMAGES
    squareDist = 50;
    coreMin = 0;
    near_join = 20;
    fillHoles = 25;
    adapt_his = 0;
    hor_factor = 1;
    
    %% ADD Y/N background subtraction
    %% ADD Y/N max-intensity projection ==> which will change if need adapthisteq
end

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
    for fileNum = 1 : numfids
        
        cd(cur_dir);
        natfnames=natsort(trialNames);
        
        %% Decide if want to load individual channels or single image
        if load_five == true
            [DAPIimage,redImage,binImage,greenImage,wholeImage] = NewFileReaderV4(trialNames, fileNum, allChoices, foldername, cur_dir);
            cd(cur_dir);

            %(1) Fibers
            nanoF_im = im2double(rgb2gray(binImage));     % Reads-in Bin
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
                %J = imerode(J, strel('disk', 1));
                nanoF_im = J;
                phaseThres = graythresh(nanoF_im);
                nanoF_im = imbinarize(nanoF_im);  % binarizes
                figure(2); imshow(nanoF_im);
            else
                nanoF_im = false(siz);
            end
            
            redImage(:, :, 2) = greenImage(:, :, 2);
            redImage(:, :, 3) = DAPIimage(:, :, 3);
            
            wholeImage = redImage;
                               
        else
            % (3) DAPI
            cd(foldername);
            % (4) Red
            filename = natfnames{fileNum};
            redImage = imread(filename);
            %O4_im = im2double(rgb2gray(redImage));
            
        end
        intensityValueDAPI = im2double(redImage(:,:,3));
        O4_original = im2double(redImage(:,:,1));
        
        cd(cur_dir);
        O4_im_ridges = O4_original;
        [O4_im, originalRed] = imageAdjust(O4_original, fillHoles, enhance);   % image adjust
        % (5) Green
        %         MBP_im = im2double(rgb2gray(greenImage));
        %         MBP_im_ridges = MBP_im;
        %         [MBP_im, originalGreen] = imageAdjust(MBP_im, fillHoles, enhance);
        
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
        [mat, objDAPI, DAPI_bw] = DAPIcount_2(intensityValueDAPI, DAPIsize, DAPImetric, enhance, siz);  % function
        
        %% Initializes struct to store everything
        c= cell(length(objDAPI), 1); % initializes Bool_W with all zeros
        [c{:}] = deal(0);
        strucMat = num2cell(mat, 2);
        s = struct('objDAPI', objDAPI', 'centerDAPI', strucMat, 'Core', cell(length(objDAPI), 1)...
            ,'CB', cell(length(objDAPI), 1), 'Fibers', cell(length(objDAPI), 1), 'Mean_Fiber_L_per_C', cell(length(objDAPI), 1), 'Bool_W', c...
            , 'im_num', c);
        
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
        [cores, cb, unass_cb, s] = O4_count(combined_im, cores, cb, siz,diameterFiber, dense, s);
        
        if verbose
            figure(70); imshow(unass_cb); title('Unassociated Objects');
        end
        
        %% Print * for DAPI and O4+
        
        wholeImage = cat(3, O4_original, zeros(size(O4_im)), intensityValueDAPI);
        
        figure(5); imshow(wholeImage); title('Output Image'); hold on;
        for Y = 1:length({s.objDAPI})
            if ~isempty(s(Y).centerDAPI) % Print DAPI
                text(s(Y).centerDAPI(1, 1),  s(Y).centerDAPI(1 ,2), '*',  'color','r' ,'Fontsize',10);   % writes "peak" besides everything
            end
            
            if ~isempty(s(Y).Core)  % Print O4+
                text(s(Y).centerDAPI(1, 1),  s(Y).centerDAPI(1 ,2), 'O4',  'color','y' ,'Fontsize',6);   % writes "peak" besides everything
                sumO4 = sumO4 + 1;
            end
        end
        
        %% (4) Sort through and find cell objects that are much too small, and set them permanently to NOT wrapped
        new_combined_im = imbinarize(unass_cb + combined_im);     % new combined_im also must include the dilated image in O4_count
        [s] = small_del_O4(new_combined_im, minLength, squareDist, siz, s);
        
        %% (5) Line seg:
        
        O4_im_ridges_adapted = O4_im;        
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
        
        
        %                 t = imbinarize(O4_im_ridges_adapted);
        %                 t = imclose(t, strel('disk', 4));
        %                 t = imcomplement(t);
        %                 D = bwdist(t);
        %                 D(D < 8) = 0;
        %                 thick = bwmorph(D, 'thicken', 5);
        %                 sub_cb = imbinarize(fibers - thick);
        %
        %                 sub_cb = fibers;
        %
        %                 clean = bwareaopen(sub_cb, 100);
        %
        %                 clean = imopen(clean, ones(10,1));
        %
        %                 % Finds orientation of each segment
        %                 cc = bwconncomp(clean);
        %                 stats = regionprops(cc, 'Orientation', 'PixelIdxList', 'MajorAxisLength');
        %
        %                 % Then sort the segments to be similar in some way???
        %                 % Everything above + 45 and below -45 ==> vertical
        %
        %                 [vert, vert_idx] = find( ([stats.Orientation] > +75  | [stats.Orientation] < -75)  & [stats.MajorAxisLength] > minLength );
        %                 vert_lines = zeros(siz);
        %                 for i = 1:length(vert_idx)
        %                     vert_lines(stats(vert_idx(i)).PixelIdxList) = 1;
        %
        %                 end
        %
        
        %                 fibers_split{Y} = vert_lines;
        %                 if Y == 1
        %                     stitched_im(1:size(I,1)/2,1:size(I,2)/2,:) = vert_lines;
        %                 elseif Y == 2
        %                     stitched_im(size(I,1)/2+1:size(I,1),1:size(I,2)/2,:) = vert_lines;
        %                 elseif Y == 3
        %                     stitched_im(1:size(I,1)/2,size(I,2)/2+1:size(I,2),:) = vert_lines;
        %                 elseif Y == 4
        %                     stitched_im(size(I,1)/2+1:size(I,1),size(I,2)/2+1:size(I,2),:) = vert_lines;
        %                 end
        %             end
        
        
        % delete the WHOLE fiber if it TOUCHES O4+ DAPI point AFTER deleting CB's already
        %             DAPI_bw = imdilate(DAPI_bw, strel('disk', 5));  % MAKE IT LARGER
        
        %% MAYBE KEEP for LOW DENSITY???
        %             if dense == 'N'
        %                 for T = 1:length(locFibers)
        %                     if  ~isempty(find(DAPI_bw(locFibers{T}), 1)) && allLengths{T} < minLength * 3
        %                         mask(locFibers{T}) = 0;   % set the ENTIRE FIBER to be nothing
        %
        %                         fibers(locFibers{T}) = 0;
        %                         fibers(mask_idx{T}) = 0;
        %                         locFibers{T} = [];
        %                         allLengths{T} = [];
        %                     end
        %                 end
        %             end
        %
        locFibers =  locFibers(~cellfun('isempty',locFibers));   % delete from the list if not a line
        allLengths = allLengths(~cellfun('isempty', allLengths));   % delete from the list if not a line
        
        %% (8) Check CBs to see if wrapped or not
        fibers_sub_cb = bwmorph(cb, 'thicken', 3);
        [locFibers, allLengths, s] = wrappingAnalysis(fibers_sub_cb, locFibers, allLengths, siz, minLength, isGreen, dense, s);
        
        %% (9) Check remaining fibers with fibers_sub_cb (real), to see if wrapped or not
        % COUNT AGAIN, with a FULL fibers_sub_cb, to get all the fibers NOT directly connected to stuff
        % and use only the REMAINING fibers (i.e. fibers(locFibers) = 0) ==> set to zero the already found ones
        fibers_sub_cb = imbinarize(combined_im - cb);  % THE REAL FIBERS_sub_cb
        [sub_locFibers, allLengths, s] = wrappingAnalysis(fibers_sub_cb, locFibers, allLengths, siz, minLength, isGreen, dense, s);
        
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
        
        %         cd(saveDirName);
        %         %For accuracy eval
        %         filename = strcat(name, '-', num2str((fileNum + 5 - 1)/5), 'obj_DAPI');
        %         save(filename, 'objDAPI');
        %
        %         filename = strcat(name, '-', num2str((fileNum + 5 - 1)/5), 'machine_DAPI');
        %         save(filename, 'wrappedDAPIR');
        %         cd(cur_dir);
        
        % For Green: %% maybe make new function?
        %         isGreen = 1;
        %         %%(1) Blobbify
        %         [blobs, blobLines, lineImage, noColocFibers] = blobbify(MBP_im, wholeImage, lengthX, lengthY, nanoF_im, phaseThres, objDAPI, diameterFiber, fillHoles, mag, isGreen, minLength, originalGreen);    % turns into blobs
        %
        %         %%(2) Houghline analysis
        %         [tmpLength, locFibersG, countedLines] = houghLineFind(lineImage, lengthX, lengthY, minLength, wholeImage);
        %         allLengthFibersG = [allLengthFibersG tmpLength];
        %
        %         %%(3) Check blobs with DAPI peaks!!!
        %         [numWrappedG, numSheathsG, wrappedDAPIG, numMBP] = wrappingAnalysis(blobs, blobLines, lineImage, noColocFibers, locFibersG, mat, objDAPI, lengthX, lengthY, tmpLength, minLength, isGreen, mag);
        %         allNumSheathsG = [allNumSheathsG numSheathsG];
        %         sumMBP = sumMBP + numMBP;
        
        % Setting variables
        numCells = length(objDAPI);   % Don't comment out
        %         numFibersG = length(allLengthFibersG);
        %         wrappedG = numWrappedG;
        %         unwrappedG = numCells - numWrappedG;
        
        numFibersG = 0;
        wrappedG = 0;
        unwrappedG = 0;
        
        lineNum = 1;
        allxy_long = cell(1,1);
        numFibersR = length(allLengthFibers);
        
        wrappedR = numWrappedR;
        unwrappedR = numCells - wrappedR;
        
        % Save image number as well:
        for Y = 1:length(s)
            s(Y).im_num = fileNum;
        end
        
        %% Print images of results
        cd(saveDirName);
        figure(5);
        filename = strcat('Result', name, num2str((fileNum + 2 - 1)/2), '_', num2str(fileNum ));
        print(filename,'-dpng')
        hold off;
        
        figure(1);
        filename = strcat('Result', name, num2str((fileNum + 2 - 1)/2),  '_', num2str(fileNum ), 'Combined_im') ;
        print(filename,'-dpng')
        hold off;
        
        % Saves image
        figure(100);
        filename = strcat('Result', name, num2str((fileNum + 2 - 1)/2),  '_', num2str(fileNum ), 'DAPI') ;
        print(filename,'-dpng')
        hold off;
        
        figure(31);
        filename = strcat('Result', name, num2str((fileNum + 2 - 1)/2),  '_', num2str(fileNum ), 'final_fibers') ;
        print(filename,'-dpng')
        hold off;
        
        figure(67); imshowpair(wholeImage, fibers); title('Filter ridges');
        filename = strcat('Result', name, num2str((fileNum + 2 - 1)/2),  '_', num2str(fileNum ), 'Filter ridges') ;
        print(filename,'-dpng'); hold off;
        
        figure(88); imshowpair(wholeImage, mask); title('Ridges to lines after sub core');  hold on;
        filename = strcat('Result', name, num2str((fileNum + 2 - 1)/2),  '_', num2str(fileNum ), 'Skeletonized ridges') ;
        print(filename,'-dpng'); hold off;
        
        
        
        %% Print to file:
        %(1) "allAnalysis.txt" is for EACH image
        
        %%if file is already open, then no need to open it again
        fprintf(fileID,'Image No. : %d              File name: %s \n', (fileNum), name);
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
    
    if batch_run == 'N'
        questTitle='More Samples?';
        start(timer('StartDelay',1,'TimerFcn',@(o,e)set(findall(0,'Tag',questTitle),'WindowStyle','normal')));
        button2 = questdlg('Another analysis?', questTitle, 'Y','N','Y','Y');
        moreTrials = button2;
        
    end
    trialNum = trialNum + 1;
    
    %% Garbage collect:
    %     clearvars -except allNames allWrappedR allWrappedG allTotalCells allTrialLengths moreTrials trialNum scale...
    %         allTrialSheathsR allTrialSheathsG allChoices foldername cur_dir allInfoInfo allSumO4 allSumMBP saveDirName cur_dir....
    %         batch_run batch_num batch_skip batch diameterFiber minLength fillHoles DAPIsize calibrate mag DAPImetric scale...
    %         sensitivity sigma nanoYN combineRG squareDist verbose enhance allTrialS allTrialMeanFLC dense save_params
    
end
%
% catch
%     cd(saveDirName);
%     fclose(fileID);
%     cd(cur_dir);
% end

%% Prompt to plot everything:
questTitle='Plot Data?';
start(timer('StartDelay',1,'TimerFcn',@(o,e)set(findall(0,'Tag',questTitle),'WindowStyle','normal')));
button3 = questdlg('Plot Data?', questTitle, 'Y','N','Y','Y');
ynPlot = button3;

ynPlot = 'Y';

if ynPlot == 'Y'
    
    defaultans = {'', 'N'};
    prompt = {'Which trial number was control?'};
    dlg_title = 'Input';
    num_lines = 1;
    answer = inputdlg(prompt,dlg_title,num_lines,defaultans);
    control_idx = str2double(cell2mat(answer(1)));
    
    %control_idx = 1;
    
    name = 'O4';
    norm_props_R = plotProps(allNames, allWrappedR, allSumO4, allTotalCells, name, control_idx, saveDirName);
    cd(cur_dir);
    
    name = 'MBP';
    norm_props_G = plotProps(allNames, allWrappedG, allSumO4, allTotalCells, name, control_idx, saveDirName);
    cd(cur_dir);
    
    %% Calls function
    plotData(allNames, allTrialLengths, allTrialSheathsR, allTrialSheathsG, saveDirName, cur_dir);
    cd(cur_dir);
    
    %% Save results
    clearvars -except allNames allTrialLengths allTrialSheathsR allTrialSheathsG allWrappedR allWrappedG allTotalCells...
        allInfoInfo allSumO4 allSumMBP saveDirName cur_dir norm_props_R norm_props_G squareDist allTrialS allTrialMeanFLC
    cd(saveDirName);
    
    
    %% Make Table
    
    allWrappedR = allWrappedR';
    allSumO4 = allSumO4';
    allTotalCells = allTotalCells';
    allWrappedG = allWrappedG';
    allNames = allNames';
    norm_props_R = norm_props_R';
    norm_props_G = norm_props_G';
    
    propW_O4 = allWrappedR./allSumO4 * 100;
    
    propW_total = allWrappedR./allTotalCells * 100;
    save('Result data');
    
    T1 = table(norm_props_R, norm_props_G, propW_O4, allWrappedR, allSumO4, allTotalCells, propW_total, allWrappedG,'RowNames', allNames)
    
    writetable(T1, 'Result_table.csv', 'WriteRowNames',true);
    
    %% Make csv files for data analysis
    name_csv = 'Result_names.csv';
    Row_Names = allNames;
    T1 = table(Row_Names);
    
    writetable(T1, name_csv, 'WriteRowNames',true);
    
    dlmwrite(name_csv, ' ', '-append');
    
    name_csv = 'Result_num_sheaths.csv';
    %dlmwrite(name_csv, 'NumWrapped,', '-append');
    for i = 1:length(allNames)
        
        row = i + 1;
        col = 2;
        
        trial_data = cell2mat(allTrialSheathsR(i));
        dlmwrite(name_csv, trial_data, '-append');
    end
    
    
    name_csv = 'Result_lengths.csv';
    for i = 1:length(allNames)
        
        row = i + 1;
        col = 2;
        
        trial_data = cell2mat(allTrialLengths(i));
        dlmwrite(name_csv, trial_data, '-append');
    end
    
    name_csv = 'Result_mean_FLC.csv';
    for i = 1:length(allNames)
        
        row = i + 1;
        col = 2;
        
        trial_data = cell2mat(allTrialMeanFLC(i));
        dlmwrite(name_csv, trial_data, '-append');
    end
    
    cd(cur_dir);
end

