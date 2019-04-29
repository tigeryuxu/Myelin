% BECAUSE currently imbinarize seems to be broken...
function [bw] = binarize_3D_otsu(im)
    thresh = graythresh(im);
    bw = im;
    bw(bw < thresh) = 0;
    bw(bw >= thresh) = 1;
end

