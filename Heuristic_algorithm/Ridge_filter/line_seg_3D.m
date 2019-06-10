function [fibers, fibers_idx, Lambda2R] = line_seg_3D(O4_im, sigma, sensitivity)

% (1) Pre-processes image by eroding ==> gets rid of "aligned" fibers
% (2) Calls ridge_filt to extract lines
% (3) Then colocs with fiber image
% (4) ***SUBTRACT cores???
%
% Inputs:
%     O4_im ==
%     nanoF_im ==
%     sigma == of gaussian filter
%     threshold == for binarization of Lambda2 image???
%     cores???
%
%     erosion amount???
%
% Outputs:
%     fibers == image of fibers
%     fibers_idx == idx of fibers

%% Ridge filt:
%O4_im = imerode(O4_im, strel('disk', 2));   % erode small lines
[hessLinesR, Lambda2R]= ridge_filt_3D(O4_im, sigma, sensitivity);




%% Colocalize to find areas of Red intersect:
% [cBin, rBin] = find(nanoF_im < 1);
% binInd = [rBin, cBin];
% [rRed, cRed] = find(hessLinesR);
% redInd = [cRed, rRed];
% inter = intersect(binInd, redInd, 'rows');
% tmpLines = zeros(siz);   % temporary array
% for test = 1: length(inter)
%     tmpLines(inter(test, 2), inter(test, 1)) = 1;   % plots all the intersecting red lines
% end

%% Save
fibers = hessLinesR;
obj = bwconncomp(fibers);
fibers_idx = obj.PixelIdxList;
figure(900); volshow(fibers, 'BackgroundColor', [0,0,0]);

end