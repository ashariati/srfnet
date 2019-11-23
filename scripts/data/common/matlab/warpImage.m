function Iw = warpImage(I, H)
%% warp the original image to generate the image after camera rotation
% input I: original image 
% input H: homography transformation which maps the points from Iw to I
% output Iw: warped image

% convert to double array
I = double(I);

% compute size
[h, w, ~] = size(I);

% pixel indices
[X, Y] = meshgrid(1:w, 1:h);
i = X(:);
j = Y(:);

% transform coordinates
x2_tilde = [i, j, ones(h*w, 1)];
x1_tilde = (H * x2_tilde')';

% scale
x1 = x1_tilde(:, 1:2) ./ x1_tilde(:, 3);

% create the meshgrid for warped (query)
Xq = reshape(x1(:, 1), h, w);
Yq = reshape(x1(:, 2), h, w);

% sample
R = uint8(interp2(X, Y, I(:, :, 1), Xq, Yq, 'cubic', 0));
G = uint8(interp2(X, Y, I(:, :, 2), Xq, Yq, 'cubic', 0));
B = uint8(interp2(X, Y, I(:, :, 3), Xq, Yq, 'cubic', intmax('uint8')));

% new image
Iw = cat(3, R, G, B);

