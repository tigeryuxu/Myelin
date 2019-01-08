%% Deconv_all

cur_dir = pwd;
cd(cur_dir);

foldername = uigetdir();   % get directory

moreTrials = 'Y';
trialNum = 1;

saveName = strcat(foldername, '_');
saveDirName = create_dir(cur_dir, saveName);
cd(cur_dir);


cd(foldername);   % switch directories
nameCat = '*.tif'
fnames = dir(nameCat);

namecell=cell(1);
for i=1:length(fnames)
        namecell{i,1}=fnames(i).name;
  
end
trialNames = namecell;
numfids = length(trialNames);   %%% divided by 5 b/c 5 files per pack currently

cd(cur_dir);
   natfnames=natsort(trialNames);
cd(foldername);
   
for fileNum = 20  : numfids
    %% Read in images
    
    cd(cur_dir);
 
    cd(foldername);
    
    
    filename = natfnames{fileNum};
    redImage = imread(filename);
    
    
    %% deconvolve
    INITPSF = ones(6, 6);
    decon = deconvblind(redImage, INITPSF);
    figure(1); imshow(decon);
    
    
    new_filename = strcat('Decon_', filename);
    cd(saveDirName);
    imwrite(decon, new_filename, 'tif');
    
end

