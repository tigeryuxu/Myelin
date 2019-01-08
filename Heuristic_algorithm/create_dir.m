function [dirName]= create_dir(cur_dir, saveName)
    % Get a list of all files and folders in this folder.
    files = dir(cur_dir);
    % Get a logical vector that tells which is a directory.
    dirFlags = [files.isdir];
    % Extract only those that are directories.
    subFolders = files(dirFlags);
    % Print folder names to command window.
    
    [path, name, ext] = fileparts(saveName);
    
    foldername = strcat(name, 'Result_1');
    numOrder = 1;
    for foldNum = 1 : length(subFolders)

        if ~isempty(findstr(subFolders(foldNum).name, foldername))
            numOrder = numOrder + 1;
            newNum = num2str(numOrder);
            if numOrder < 10
                  foldername(end) = newNum;
            else
                foldername(end- 1) = '_';
                foldername(end + 1) = ' ';
                foldername(end - 1 : end) = newNum;
            end
        end        
    end
    %mkdir(foldername);
    
    dirName = foldername;
    
end
       