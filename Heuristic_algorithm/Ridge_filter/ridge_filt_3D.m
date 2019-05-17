function [bw, Lambda2] = ridge_filt(img, sigma, sensitivity)

% Calls methods to compute the hessian matrix and solve for the eigenvalues of the image
% The resulting Lambda2 is used as filtered ridges and is binarized and cleaned
% 
% Input:
%     img ==
%     sigma == 
%     THRESHOLD (optional???) == 
%     
% Output:
%     bw == binarized and thresholded eigenvalue matrix
%     Lambda2 == original raw eigenvalue matrix

% FIND EIGENVALUES of HESSIAN matrix

%tmp_img = img /255;
tmp_img = img;
[Dxx, Dyy, Dzz, Dxy, Dxz, Dyz]= Hessian3D(tmp_img, sigma);
[Lambda1,Lambda2,Lambda3,Vx,Vy,Vz]=eig3volume(Dxx,Dxy,Dxz,Dyy,Dyz,Dzz);
% THRESHOLD
lineThresh = thresh_Lambda(Lambda3);
lineThresh = lineThresh + lineThresh * sensitivity;

bw = binarize_3D_lambda(Lambda3, lineThresh);

% CLEAN
%bw = bwareaopen(~bw, 50);

end