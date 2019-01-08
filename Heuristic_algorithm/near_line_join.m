function [locFibers, s] = near_line_join(locFibers, max_thresh, siz, verbose, s)

% Joins remaining lines to nearest WRAPPED cell body by adding
% the line to the "struct" of the wrapped cell
%
% (1) Finds bwboundary of all WRAPPED cells + fibers in struct
% (2) Then loops through each fiber, and calculates the distance of the fiber boundary to the cell boundary
%         ==> uses pythagoras or norm() function
% (3) The smallest pythagoras ==> the nearest cell
%
% OR
%
% (1) Plot the line on a tmp array
% (2) Then invert, and do bwdist transform
% (3) Then cycle through each DAPI point, and use the indices to look at an area of the bwdist matrix
% (4) find the min value in that area and save it
% (5) compare the min values for each DAPI blob region
% (6) if MIN value is < MAX threshold, then we add the line to the STRUCT
%
% Input:
%     struct ==> contains WRAPPED cell indices + the associated fibers for each cell
%
%     cb_idx ==> all the DAPI blobs (not just the wrapped ones)
%
%     MAX threshold (optional) ==> default is 1000 pixels???
%
%     locFibers ==> contains the remaining fibers
%
% Output:
%     struct ==> that now has the additional fibers

debugLocFibers = locFibers;

%% Loop through all the forgotten fibers
for k = 1:length(locFibers)
    curFiber = locFibers{k};
    if ~isempty(curFiber)
        tmp = zeros(siz);
        tmp(curFiber) = 1;  % creates image with the fiber
        tmp = imbinarize(tmp);
        %tmp = imcomplement(tmp);
        tmp = bwdist(tmp);    % Gets distance transform matrix
        
        allMin = [];
        allIdx = [];
        
        %% Loop through all the fibers for each cell
        for i = 1:length({s.Fibers})
            curFibers = s(i).Fibers;
            for N = 1:length(curFibers)
                if ~isempty(curFibers{N})   % if cell was identified as wrapped
                    allMin = [allMin min(tmp(curFibers{N}))];
                    allIdx = [allIdx i];
                end
            end
        end
        
        % Add the fiber to the cell with minimum distance:
        [val, minIdx] = min(allMin);
        cellIdx = allIdx(minIdx);
        
        % if not too far away
        if val < max_thresh
            s(cellIdx).Fibers{end + 1} = curFiber;
            locFibers{k} = [];
        end
    end
end

%% Plot all the fibers for EACH cell struct
plotNumbers = [];
colorTmp = zeros(siz);
for i = 1:length({s.objDAPI})
    colors = rand(1, 3);  % colors
    number = [];
    
    curFibers = s(i).Fibers;
    if ~isempty(curFibers)   % if cell was identified as wrapped
        
        % Loop through all the current fibers
        for N = 1:length(curFibers)
            
            colorTmp(curFibers{N}) = colors(1);
            
            [x, y] = ind2sub(siz, curFibers{N}(1));
            
            number = [i, floor(x), floor(y)];
            plotNumbers = [plotNumbers; number];
        end
    end
end
labelled = label2rgb(round(colorTmp*255));
figure(31); imshow(labelled); hold on;
title('near line join');

if ~isempty(plotNumbers)
    for L = 1:length(plotNumbers(:, 1))
        text(plotNumbers(L, 3), plotNumbers(L, 2), num2str(plotNumbers(L, 1)),  'color','k' ,'Fontsize',10);   % writes "peak" besides everything
    end
end


end