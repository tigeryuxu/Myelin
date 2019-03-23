%% Main function to run heuristic algorithm
cur_dir = pwd;
addpath(strcat(cur_dir, '\IO_func'))  % adds path to functions
cd(cur_dir);
saveDirName = uigetdir();   % get directory
rawDataDirName = uigetdir();   % get directory
batch_numFiles = []; minLength = 12;
scale = 0.454;

load_five = 0;  % set to 1 if loading sets of 5 images

%% FOR DARYAN IMAGE ==> change "load_five" and change "size limits"

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


%% Load raw data names
cd(cur_dir)
cd(rawDataDirName)
nameCat_raw = '*tif';
fnames_raw = dir(nameCat_raw);
namecell_raw=cell(1);
idx = 1;
for i=1:length(fnames_raw)
    namecell_raw{idx,1}=fnames_raw(i).name;
    idx = idx + 1;
end
trialNames_raw = namecell_raw;
cd(cur_dir);
trialNames_raw = natsort(trialNames_raw);
cd(rawDataDirName);
numfids_raw = length(trialNames_raw) - 1;   %%% divided by 5 b/c 5 files per pack currently
mkdir('UNet output');

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

%% if load_five
raw_counter = 1;
for fileNum = 1 : numfids
    
    allNumSheathsR = [];
    allLengthFibersR = [];
    allMeanLPC = [];
    allLog = [];
    all_area_per_cell = [];
    num_O4_individual = 0;
    num_sheaths_individual = 0;
    
    cd(saveDirName);
    filename = trialNames{fileNum};
    s = load(filename);
    s = s.allS;
    
    cd(rawDataDirName);
    if load_five == 1
        
        % +2 ==> is red, +3 ==> is DAPI, +4 ==> is green
        
        filename_raw = trialNames_raw(raw_counter + 3);
        DAPIimage = imread(filename_raw{1});
        DAPIimage = DAPIimage(:, :, 3);
        
        filename_raw = trialNames_raw(raw_counter + 4);
        greenImage = imread(filename_raw{1});
        redImage = greenImage(:, :, 2);
        
        %filename_raw = trialNames_raw(raw_counter + 2);
        %redImage = imread(filename_raw{1});
        %redImage = redImage(:, :, 1);
        raw_counter = raw_counter + 5;
        
        wholeImage = cat(3, redImage, zeros(size(redImage)), DAPIimage);
        
    else
        raw_counter = fileNum;
        filename_raw = trialNames_raw(raw_counter);
        wholeImage = imread(filename_raw{1});
        
        wholeImage(:, :, 2) = zeros(size(wholeImage(:, :, 2)));
        
    end
    
    
    size_red = size(wholeImage);
    square_cut_h = size_red(1);
    square_cut_w = size_red(2);
    
    
    % makes the image smaller, else runs out of RAM
    if square_cut_h > 50000
        square_cut_h = 5000;
    end
    if square_cut_w > 65000
        square_cut_w = 6500;
    end
    
    wholeImage = wholeImage(1:square_cut_h, 1: square_cut_w, :);
    
    
    
    
    if isempty(s)
        all_individual_trials = [all_individual_trials; [0,  num_O4_individual,num_sheaths_individual]];
    else
        
        size_im = s(1).im_size;
        height = size_im(1);
        width = size_im(2);
        
        % Count allNumSheaths and allLengths  ***using LENGTH OF SKELETON
        all_area_per_cell = s(1).AreaOverall;
        for N = 1:length({s.objDAPI})
            arrayfun(@cla,findall(0,'type','axes'));
            if isfield(s, 'O4_bool') && s(N).O4_bool
                numO4 = numO4 + 1;
                num_O4_individual = num_O4_individual + 1;
            end
            
            
            % Put the fibers into a tmp array so can find MajorAxisLength
            fibers_cell = [];
            tmp = zeros([height, width]);
            for Y = 1:length(s(N).Fibers)
                tmp(s(N).Fibers{Y}) = 1;
            end
            
            %tmp = imdilate(tmp, strel('disk', 10));
            [B,L] = bwboundaries(tmp, 'noholes');
            vv = regionprops(L, 'MajorAxisLength', 'PixelIdxList');
            num_sheaths = 0;
            
            new_vv = cell(0);
            %tmp_vv = vv;
            for Y = 1:length(vv)
                len = vv(Y).MajorAxisLength;
                if len > minLength / scale
                    new_vv{end + 1} = len;
                else
                    vv(Y).PixelIdxList = [];
                end
            end
            
            
            %% Re-make the tmp mask
            tmp = zeros([height, width]);
            for Y = 1:length(vv)
                if ~isempty(vv(Y).PixelIdxList)
                    tmp(vv(Y).PixelIdxList) = 2;
                end
            end
            
            %% use DAPI point as center to crop images
            DAPI_idx = s(N).objDAPI;
            DAPI_center = s(N).centerDAPI;
            
            x_size = round(DAPI_center(1));
            y_size = round(DAPI_center(2));
            
            total_length_x = 640;
            total_length_y = 1024;
            
            x_left = x_size - total_length_x / 2;
            x_right = x_size + total_length_x / 2;
            
            % adaptive cropping for width (x-axis)
            if x_left <= 0
                x_right = x_right + abs(x_left) + 1;
                x_left = 1;
                
            elseif x_right > width
                x_left = x_left - (x_right - width);
                x_right = width;
            end
            
            % adaptive cropping for height (y-axis)
            y_top = y_size - total_length_y / 2;
            y_bottom = y_size + total_length_y / 2;
            if y_top <= 0
                y_bottom = y_bottom + abs(y_top) + 1;
                y_top = 1;
                
            elseif y_bottom > height
                y_top = y_top - (y_bottom - height);
                y_bottom = height;
            end
            
            % Final check to see if sizes are correct
            if (x_right - x_left) ~= 640 || (y_bottom - y_top) ~= 1024
                break;
                t = 'ERROR, ERROR, ERROR'
            end
            
            
            %% Insert DAPI mask into wholeImage BEFORE cropping
            blank_DAPI = zeros([height, width]);
            blank_DAPI(DAPI_idx) = 255;
            
            wholeImage(:, :, 2) = blank_DAPI;
            
            %% CROP
            
            
            %crop_tmp = imcrop(tmp, [x_left y_top total_length_x-1 total_length_y-1]);
            %crop_wholeImage = imcrop(wholeImage, [x_left y_top total_length_x-1 total_length_y-1]);
            
            crop_tmp = tmp(y_top:y_bottom - 1, x_left:x_right - 1);
            crop_wholeImage = wholeImage(y_top:y_bottom - 1, x_left:x_right - 1, :);
            
            
            %% For debug
            figure(1); imshow(crop_tmp);
            figure(2); imshow(crop_wholeImage);
            
            
            
            
            %% SAVE CROP output
            
            if s(N).Bool_W == 1 && ~isempty(new_vv)
                output = '_pos';
            else
                output = '_neg';
            end
            
            cd('UNet output');            
            filename_raw = erase(filename_raw, '.tif');
            filename_raw = erase(filename_raw, 'Extended Depth of Focus');
            filename_raw = erase(filename_raw, 'Image Export');
            f_n = strcat(filename_raw,'_', sprintf('%06d',N), output, '_truth.tif');
            imwrite(crop_tmp, f_n{1});
            f_n = strcat(filename_raw, '_', sprintf('%06d',N), output, '_input.tif');
            imwrite(crop_wholeImage, f_n{1});

            cd(rawDataDirName);
            
        end
    end

end
