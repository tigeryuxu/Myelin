%% ADD TO: ==> "mainDAPI_mask_identify.m"
%% COLOC with existing hand counted DAPI, and exclude them (DAPI == tmp)
%     newObjDAPI = objDAPI;
%     for Y = 1:length(objDAPI)
%         curDAPI = objDAPI{Y};
%         if ~isempty(find(tmp(curDAPI), 1))
%             newObjDAPI{Y} = [];
%             %fprintf('in')
%         end
%     end
%
%     % Then re-create DAPI_bw, but make the numbering start from the end
%     % of the last numbering system (which augments by 2 as well)
%     last = max(tmp_fibers(:));
%     first_num = last + 1;
%
%     new_DAPI_bw = zeros(siz);
%     for Y = 1:length(newObjDAPI)
%         curDAPI = newObjDAPI{Y};
%         new_DAPI_bw(curDAPI) = first_num;
%         first_num = first_num + 2;
%     end
%
%      % Then add these new DAPI points to the existing tmp image, but with value of 10000)
%
%
%      %new_DAPI_bw(new_DAPI_bw > 0) = 1421;
%
%      combined = tmp + uint16(new_DAPI_bw);
%      filename = strcat(first{1}, '_ALL_DAPI_mask.tif');
%      cd(cur_dir);
%      imwrite(combined, filename);

%
% name = 'Mask_uFNet-03_2_s11z5c1+2+3_99_ALL_DAPI_mask.tif';
% all = imread(name);
% name = 'Mask_uFNet-03_2_s11z5c1+2+3_99_DAPI_mask.tif';
% DAPI = imread(name);
% name = 'Mask_uFNet-03_2_s11z5c1+2+3_99_fibers_mask.tif';
% fibers = imread(name);
%name = 'Mask_uFNet-03_2_s11z5c1+2+3_99.tif';
%im = imread(name);
cur_dir = pwd;
foldername = uigetdir();   % get directory

cd(foldername)
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

    cd(foldername)
    name = trialNames{fileNum};
    im = imread(name);
    
    cd(cur_dir);
    tmp = zeros(size(im), 'uint16');
    %% Create mask_array of DAPI only
    for i = 1: 2 : max(im(:))
        val =  im(find(im==i));
        if ~isempty(val)
            tmp(find(im == i)) = val(1);
        end
    end
    %figure; imshow(tmp, []);
    filename = strsplit(name, '.');
    first = filename(1);
    filename = strcat(first{1}, '_DAPI_mask.tif');
    imwrite(tmp, filename);
    
    %% Create mask_array of Fibers only
    tmp_fibers = zeros(size(im), 'uint16');
    for i = 1: 2 : max(im(:))
        val =  im(find(im==i + 1));
        erode = zeros(size(im), 'uint16');
        if ~isempty(val)
            tmp_fibers(find(im == i + 1)) = val(1);
            %erode(find(im==i + 1)) = val(1);
            
            %new = imerode(erode, strel(ones(1, 3)));
            %tmp_fibers = tmp_fibers + new;
            
        end
    end
    %figure; imshow(tmp_fibers, []);
    filename = strsplit(name, '.');
    first = filename(1);
    filename = strcat(first{1}, '_fibers_mask.tif');
    imwrite(tmp_fibers, filename);
    
end


%tmp_fibers = zeros(size(stitched_im), 'uint16');
