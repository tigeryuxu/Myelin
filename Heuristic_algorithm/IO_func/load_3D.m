%fname = 'my_file_with_lots_of_images.tif';
info = imfinfo(filename_raw);
num_images = numel(info);
im_size = size(A);
gray_scale_size = im_size(1:2);
green_3D = zeros([gray_scale_size, num_images]);
red_3D = zeros([gray_scale_size, num_images]);
for k = 1:num_images
    A = imread(filename_raw, k, 'Info', info);
    % ... Do something with image A ...
    figure(1); imshow(A);
    red = A(:, :, 1);
    green = A(:, :, 2);

    green_3D(:, :, k) = red;
    red_3D(:, :, k) = green;

   
end