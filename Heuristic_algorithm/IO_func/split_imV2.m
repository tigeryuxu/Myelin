function [all_images_cell] = split_imV2(im, blockSizeR, blockSizeC)
    
    im_mat = im;
    [rows, columns, numberOfColorBands]=size(im_mat);
    
    % Figure out the size of each block in rows.
    % Most will be blockSizeR but there may be a remainder amount of less than that.
    wholeBlockRows = floor(rows / blockSizeR);
    blockVectorR = [blockSizeR * ones(1, wholeBlockRows), rem(rows, blockSizeR)];
    % Figure out the size of each block in columns.
    wholeBlockCols = floor(columns / blockSizeC);
    blockVectorC = [blockSizeC * ones(1, wholeBlockCols), rem(columns, blockSizeC)];
    
    % Create the cell array, all_images_cell.
    % Each cell (except for the remainder cells at the end of the image)
    % in the array contains a blockSizeR by blockSizeC by 3 color array.
    % This line is where the image is actually divided up into blocks.
    if numberOfColorBands > 1
        % It's a color image.
        all_images_cell = mat2cell(im_mat, blockVectorR, blockVectorC, numberOfColorBands);
    else
        all_images_cell = mat2cell(im_mat, blockVectorR, blockVectorC);
    end
    
%     % Now display all the blocks.
%     plotIndex = 1;
%     numPlotsR = size(all_images_cell, 1);
%     numPlotsC = size(all_images_cell, 2);
%     for r = 1 : numPlotsR
%         for c = 1 : numPlotsC
%             fprintf('plotindex = %d,   c=%d, r=%d\n', plotIndex, c, r);
%             % Specify the location for display of the image.
%             subplot(numPlotsR, numPlotsC, plotIndex);
%             % Extract the numerical array out of the cell
%             rgbBlock = all_images_cell{r,c};
%             imshow(rgbBlock); % Could call imshow(all_images_cell{r,c}) if you wanted to.
%             [rowsB columnsB numberOfColorBandsB] = size(rgbBlock);
%             % Make the caption the block number.
%             caption = sprintf('Block #%d of %d\n%d rows by %d columns', ...
%                 plotIndex, numPlotsR*numPlotsC, rowsB, columnsB);
%             title(caption);
%             drawnow;
%             % Increment the subplot to the next location.
%             plotIndex = plotIndex + 1;
%         end
%     end
    

end
