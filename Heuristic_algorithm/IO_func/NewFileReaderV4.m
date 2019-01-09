
function [DAPIimage,redImage,binImage,greenImage,wholeImage]= NewFileReaderV4(trialNames, fileNum, allChoices, foldername, cur_dir)

cd(foldername); % switch to folder

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
            elseif mm==2
                filename = natfnames(fileNum + mmm-1);
                redImage = imread(filename{1,1});
            elseif mm==3
                filename = natfnames(fileNum + mmm-1);
                binImage = imread(filename{1,1});
            elseif mm==4
                filename = natfnames(fileNum + mmm-1);
                greenImage = imread(filename{1,1});
            elseif mm==5
                filename = natfnames(fileNum + mmm-1);
                wholeImage = imread(filename{1,1});
            end
        end
    end
end
end

