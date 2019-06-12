 %mex eig3volume.c
 
%fname = 'my_file_with_lots_of_images.tif';
info = imfinfo(filename_raw);
num_images = numel(info);
im_size = [info(1).Height, info(1).Width];
gray_scale_size = im_size(1:2);
green_3D = zeros([gray_scale_size, num_images]);
red_3D = zeros([gray_scale_size, num_images]);
for k = 1:num_images
    A = imread(filename_raw, k, 'Info', info);
    % ... Do something with image A ...
    figure(1); imshow(A);
    red = A(:, :, 1);
    green = A(:, :, 2);
    
    red_3D(:, :, k) = im2double(red);
    green_3D(:, :, k) = im2double(green);
end

iptsetpref('VolumeViewerUseHardware',false);   % HAVE TO USE THIS b/c problem with openGL currently
%volumeViewer(red_3D);
figure(400); volshow(red_3D,  'BackgroundColor', [0,0,0]);
figure(401); volshow(green_3D,  'BackgroundColor', [0,0,0]);



%bw = imbinarize(red_3D, 30);
red_3D = imgaussfilt3(red_3D,0.5);
bw = binarize_3D_otsu(red_3D);
figure(); volshow(bw, 'BackgroundColor', [0,0,0]);


green_3D(green_3D < 70/255) = 0;
figure(); volshow(green_3D);
green_3D = imgaussfilt3(green_3D,0.5);
bw = binarize_3D_otsu(green_3D);
figure(); volshow(bw, 'BackgroundColor', [0,0,0]);



%% Try out ridge-filter in 3D
sigma = 2;
%[hxx,hxy, hyy] = Hessian3D(red_3D, sigma);
%[Lambda1, Lambda2, Ix, Iy] = eig3volume(hxx, hxy, hyy);

[Dxx, Dyy, Dzz, Dxy, Dxz, Dyz] = Hessian3D(red_3D,1);
[Lambda1,Lambda2,Lambda3,Vx,Vy,Vz]=eig3volume(Dxx,Dxy,Dxz,Dyy,Dyz,Dzz);

sensitivity = 0;
lineThresh = thresh_Lambda(Lambda3);
lineThresh = lineThresh + lineThresh * sensitivity;
bw = binarize_3D_lambda(Lambda3, lineThresh);
figure(); volshow(bw, 'BackgroundColor', [0,0,0]);

fig_num = 10;
vol_to_show_stack(bw, num_images, fig_num);

cc = bwconncomp(bw);
labelled = zeros(size(bw));
for i = 1:length(cc(1).PixelIdxList)
    cur_idx = cc(1).PixelIdxList;
   labelled(cur_idx{i}) = i; 
end
figure(); volshow(labelled, 'BackgroundColor', [0,0,0]);


iptsetpref('VolumeViewerUseHardware',true)



