function [] = save_internode_data_3D(im, saveDirName)
    cc = bwconncomp(im);
    vv = regionprops3(cc, 'PrincipalAxisLength');
    
    % GET PIXEL-LENGTH
    L = [];
    if ~isempty(cc.PixelIdxList)
        for i=1:length(cc.PixelIdxList)
           L = [L, length(cc.PixelIdxList{i})]; 
        end

        if ~isempty(vv)
            %L = vv.PrincipalAxisLength(:, 1);   L = L';
            if isempty(L)   L = 0;  end
            dlmwrite(strcat('internodes', saveDirName, '.csv'), L, '-append');

        else
            L = 0;
            dlmwrite(strcat('internodes', saveDirName, '.csv'), L, '-append'); 
        end
    end
end