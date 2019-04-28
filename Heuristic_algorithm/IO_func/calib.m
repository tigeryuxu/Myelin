function [sensitivity] =  calib(sigma, foldername, cur_dir)

%Calibrates the sensitivity level as well as other attributes for the O4 file

cd(foldername);

filename = uigetfile('*.*', 'Choose Cy3 calibration file');
redImage = imread(filename);
cd(cur_dir);
O4_im = im2double(rgb2gray(redImage));
O4_im_ridges = O4_im * 255;

%% Ridge-filt code, but compare with THRESH_TOOL
% FIND EIGENVALUES of HESSIAN matrix
[hxx, hxy, hyy]= Hessian2D(O4_im_ridges, sigma);
[Lambda1,Lambda2,Ix,Iy]=eig2image(hxx,hxy,hyy);

% THRESHOLD
autoLineThresh = thresh_Lambda(Lambda2);
figure(1); imshow(O4_im/255);
userLineThresh = thresh_tool(Lambda2);

% Calculate sensitivity:
sensitivity = (autoLineThresh - userLineThresh) / autoLineThresh;
sensitivity = abs(sensitivity)

end