function [DAPIimage,redImage,binImage,greenImage,wholeImage, im_size]= NewFileReaderV4(trialNames, fileNum, allChoices, foldername, cur_dir)

cd(foldername); % switch to folder

elim_none = cell(0);
for i = 1:length(allChoices)
    if ~strcmp(allChoices{i}, 'None')
        elim_none{end + 1} = allChoices{i};
    end
end
allChoices = elim_none;


[rallchoices,callchoices]=size(allChoices);

types=cell(5,1);
types(1,1)={'DAPI'};
types(2,1)={'Red Channel'};
types(3,1)={'Fibers'};
types(4,1)={'Green Field'};
types(5,1)={'whole'};

% namecell=cell(length(fnames),1); %%must be out of function
% for i=1:length(fnames)
%     namecell{i,1}=fnames(i).name ;
% end

cd(cur_dir);natfnames=natsort(trialNames);
cd(foldername);

for mmm=1:callchoices
    filetype=allChoices{1,mmm};
    for mm=1:5
        tf= strcmp(filetype,types(mm,1));
        if tf==1
            if mm==1
                filename = natfnames(fileNum + mmm-1);
                DAPIimage = imread(filename{1,1});
                im_size = size(DAPIimage);
            elseif mm==2
                filename = natfnames(fileNum + mmm-1);
                redImage = imread(filename{1,1});
                im_size = size(redImage);
            elseif mm==3
                filename = natfnames(fileNum + mmm-1);
                binImage = imread(filename{1,1});
                im_size = size(binImage);
            elseif mm==4
                filename = natfnames(fileNum + mmm-1);
                greenImage = imread(filename{1,1});
                im_size = size(greenImage);
            elseif mm==5
                filename = natfnames(fileNum + mmm-1);
                wholeImage = imread(filename{1,1});
                im_size = size(wholeImage);
            end
            %         elseif filetype == 'None'   % if filetype == 'None'
            %             if mmm==1
            %                 DAPIimage = [];
            %             elseif mmm==2
            %                 redImage = [];
            %             elseif mmm==3
            %                 binImage = [];
            %             elseif mmm==4
            %                 greenImage = [];
            %             elseif mmm==5
            %                 wholeImage = [];
            %             end
        end
    end
end

if exist('redImage') == 0
    redImage = [];
end
if  exist('greenImage') == 0
    greenImage = [];
end
if  exist('DAPIimage') == 0
    DAPIimage = [];
end
if  exist('wholeImage') == 0
    wholeImage = [];
end
if  exist('binImage') == 0
    binImage = [];
end


end

