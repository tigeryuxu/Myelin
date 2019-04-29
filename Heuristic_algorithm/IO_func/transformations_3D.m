% BECAUSE currently imbinarize seems to be broken...
function [bw] = binarize_3D_otsu(im)
    thresh = graythresh(im);
    bw = im;
    bw(bw < thresh) = 0;
    bw(bw >= thresh) = 1;
end

function [bw] = binarize_3D_lambda(im, thresh)
    %thresh = graythresh(im);
    bw = im;
    bw(bw >= thresh) = 0;
    bw(bw < thresh) = 1;
end

function [] = vol_to_show_stack(vol, num_images, fig_num)
    for k = 1:num_images
        im = vol(:, :, k);
        figure(fig_num); imshow(im,[]);
    end
end
