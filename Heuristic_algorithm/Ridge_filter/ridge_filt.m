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

tmp_img = img /255;

[hxx, hxy, hyy]= Hessian2D(tmp_img, sigma);
[Lambda1,Lambda2,Ix,Iy]=eig2image(hxx,hxy,hyy);
% THRESHOLD
lineThresh = thresh_Lambda(Lambda2);
lineThresh = lineThresh + lineThresh * sensitivity;

bw = imbinarize(Lambda2, lineThresh);

% CLEAN
bw = bwareaopen(~bw, 50);

end