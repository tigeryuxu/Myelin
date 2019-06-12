
function [] = vol_to_show_stack(vol, num_images, fig_num)
    for k = 1:num_images
        im = vol(:, :, k);
        figure(fig_num); imshow(im,[]);
    end
end