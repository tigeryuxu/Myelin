opengl hardware;
close all;

cur_dir = pwd;
cd(cur_dir);

%% For stats
allWrappedR = []; allWrappedG = []; allTotalCells = [];
allNames = cell(0); allTrialLengths = cell(0);  allTrialSheathsR = cell(0); allTrialSheathsG = cell(0); allInfoInfo = cell(0);
allTrialS = cell(0); allTrialMeanFLC = cell(0);
allSumO4 = []; allSumMBP = [];
full_table = [];

foldername = uigetdir();   % get directory

%% GET MASK names
DAPI_mask_folder = uigetdir();   % get directory
cd(DAPI_mask_folder);
nameCat = '*tif';
fnames = dir(nameCat);
namecell=cell(1);
idx = 1;
for i=1:length(fnames)
    namecell{idx,1}=fnames(i).name;
    idx = idx + 1;
end
trialNames_mask = namecell;
cd(cur_dir);
mask_counter = 1;

allChoices = choosedialog2();   %% read in choices  %%% SWITCH TO FROM GUI.m

moreTrials = 'Y';
trialNum = 1;

saveName = strcat(foldername, '_');
saveDirName = create_dir(cur_dir, saveName);   % creates directory
mkdir(saveDirName);

cd(foldername);   % switch directories
nameCat = '*tif';
fnames = dir(nameCat);

namecell=cell(1);
idx = 1;
for i=1:length(fnames)
    namecell{idx,1}=fnames(i).name;
    idx = idx + 1;
end
trialNames = namecell;
numfids = length(trialNames);   %%% divided by 5 b/c 5 files per pack currently

%% Read in images
for fileNum = 1 : numfids
    
    cd(cur_dir);
    natfnames=natsort(trialNames);
    
    % (3) DAPI
    cd(foldername);
    filename = natfnames{fileNum};
    wholeImage = imread(filename);
    
    DAPI_im = wholeImage(:, :,  3);
    intensityValueDAPI = im2double(DAPI_im);
    
    redImage = wholeImage(:, :, 1);
    redImage = im2double(redImage);
    
    fillHoles = 25;
    enhance = 'Y';
    O4_size = 20;
    cd(cur_dir);
    [O4_im, originalRed] = imageAdjust(redImage, fillHoles, enhance);   % image adjust
    
    %% Get rid of small pieces of O4_im
    %    open_O4 = imopen(O4_im, strel('disk', 5));  % ALL O4
    %    open_blobs =  imclose(O4_im, strel('disk', 20));
    %    open_blobs = imopen(open_blobs, strel('disk', 15));   % CANDIDATES FOR ANALYSIS
    
    siz = size(intensityValueDAPI);
    %% (1) Find peaks for DAPI
    DAPIsize = 20;
    DAPImetric = 0.20;
    enhance = 'N';
    cd(cur_dir);
    [mat, objDAPI, DAPI_bw] = DAPIcount_2(intensityValueDAPI, DAPIsize, DAPImetric, enhance, siz);  % function
    cd(foldername);
    
    c= cell(length(objDAPI), 1); % initializes Bool_W with all zeros
    [c{:}] = deal(0);
    strucMat = num2cell(mat, 2);
    s = struct('objDAPI', objDAPI', 'centerDAPI', strucMat, 'Core', cell(length(objDAPI), 1)...
        ,'CB', cell(length(objDAPI), 1), 'Fibers', cell(length(objDAPI), 1), 'Mean_Fiber_L_per_C', cell(length(objDAPI), 1), 'Bool_W', c...
        , 'im_num', c);
    
    cd(cur_dir);
    
    %% (2) Extract cores
    coreMin = 20;
    diameterFiber = 10;
    percentDilate = 2;
    [s] = reg_core_filt(O4_im, diameterFiber, siz, percentDilate, s);  % small cores
    [cb, no_dilate_cb, s] = cell_body_filt(O4_im, diameterFiber, siz, coreMin, s);        % cell body
    
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
    
    
    %% Colocalize to find areas of Red intersect:
    %     [cBin, rBin] = find(DAPI_bw);
    %     binInd = [rBin, cBin];
    %     [rRed, cRed] = find(open_blobs);
    %     redInd = [cRed, rRed];
    %     inter = intersect(binInd, redInd, 'rows');
    %     tmpLines = zeros(siz);   % temporary array
    %     for test = 1: length(inter)
    %         tmpLines(inter(test, 2), inter(test, 1)) = 1;   % plots all the intersecting red lines
    %     end
    
    %% can also dilate O4 to make it more lenient
    [B,L] = bwboundaries(tmpLines, 'noholes');
    new_objDAPI = regionprops(L,'Centroid', 'PixelIdxList'); %%%***good way to get info about region!!!
    
    DAPI_O4 = wholeImage;
    DAPI_O4(:, :, 2) = zeros(size(redImage));
    %figure(5); imshow(DAPI_O4); title('Output Image'); hold on;
    for Y = 1:length(new_objDAPI)
        if ~isempty(new_objDAPI(Y).Centroid) % Print DAPI
            centerDAPI = new_objDAPI(Y).Centroid;
            %text(centerDAPI(1, 1),  centerDAPI(1 ,2), '*',  'color','g' ,'Fontsize',8);   % writes "peak" besides everything
        end
    end
    len_cand = length(new_objDAPI);
    
    
    %% Print * for DAPI and O4+
    %wholeImage = cat(3, redImage, zeros(siz), intensityValueDAPI);
    %figure(5); imshow(wholeImage); title('Output Image'); hold on;
    
    % Save image number as well:
    image_number = (fileNum);
    cd(cur_dir);
    %% Print images of CANDIDATES
    cd(saveDirName);
    split_n = strsplit(filename, '.');
    filename = strcat(split_n{1}, '_CANDIDATES_mask.tif');
    imwrite(tmpLines, filename);
    
    %% Colocalize to find areas of Red intersect:
    [cBin, rBin] = find(DAPI_bw);
    binInd = [rBin, cBin];
    [rRed, cRed] = find(open_O4);
    redInd = [cRed, rRed];
    inter = intersect(binInd, redInd, 'rows');
    tmpLines = zeros(siz);   % temporary array
    for test = 1: length(inter)
        tmpLines(inter(test, 2), inter(test, 1)) = 1;   % plots all the intersecting red lines
    end
    
    %% can also dilate O4 to make it more lenient
    [B,L] = bwboundaries(tmpLines, 'noholes');
    new_objDAPI = regionprops(L,'Centroid', 'PixelIdxList'); %%%***good way to get info about region!!!
    
    %     DAPI_O4 = wholeImage;
    %     DAPI_O4(:, :, 2) = zeros(size(redImage));
    %     figure(5); imshow(DAPI_O4); title('Output Image'); hold on;
    %     for Y = 1:length(new_objDAPI)
    %         if ~isempty(new_objDAPI(Y).Centroid) % Print DAPI
    %             centerDAPI = new_objDAPI(Y).Centroid;
    %             text(centerDAPI(1, 1),  centerDAPI(1 ,2), '*',  'color','g' ,'Fontsize',8);   % writes "peak" besides everything
    %         end
    %     end
    len_O4 = length(new_objDAPI);
    
    
    %% SHOULD ALSO KEEP NUMBER OF O4+ cells as count
    table = zeros(1, 3);
    table(1) = length(objDAPI);
    table(2) = length(len_cand);
    table(3) = length(len_O4);
    full_table = [full_table; table];
end

%% save full_table

