
function [bw] = binarize_3D_lambda(im, thresh)
    %thresh = graythresh(im);
    bw = im;
    bw(bw >= thresh) = 0;
    bw(bw < thresh) = 1;
end
