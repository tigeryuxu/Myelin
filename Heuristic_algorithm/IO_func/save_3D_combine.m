function [] = save_3D_combine(red, green, blue, filename_save, im_size)

num_images = im_size(3);
for k = 1:num_images
    red_2D = red(:, :, k);
    green_2D = green(:, :, k);
    blue_2D = blue(:, :, k);
    
    combined = cat(3, red_2D, green_2D, blue_2D);
    
    %figure(888); imshow(combined);
    imwrite(combined, filename_save, 'writemode', 'append', 'Compression','none')
end





