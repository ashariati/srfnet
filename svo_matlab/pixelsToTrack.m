% NOTE: we are not returning subpixel locations since in our super resolution
% network, we can hardly expect the gradient to provide sub-pixel accuracy w.r.t. 
% corner positions
%
function [corners, edgelets] = pixelsToTrack(Ix, Iy)

% intermediary values
Ix2 = Ix .* Ix;
Iy2 = Iy .* Iy;
Ixy = Ix .* Iy;

% image magnitude
Im = sqrt(Ix2 + Iy2);

% edgelets
edgelets = (Im > 400);

% shi-tomasi corners
score = stCornerScore(Ix2, Iy2, Ixy);
loc = stCornerLocations(score);

% subpixel locations
% subpixLoc = subPixelLocation(score, loc);
% subpixVal = computeMetric(score, subpixLoc);

% pack up
% corners = cornerPoints(subpixLoc, 'Metric', subpixVal);
corners = cornerPoints(loc);

end

function score = stCornerScore(A, B, C)

% crop the valid gradients
A = A(2:end-1,2:end-1);
B = B(2:end-1,2:end-1);
C = C(2:end-1,2:end-1);

% gaussian weighting
Fg = fspecial('gaussian', 5, 5/3);
A = imfilter(A, Fg, 'replicate', 'full', 'conv');
B = imfilter(B, Fg, 'replicate', 'full', 'conv');
C = imfilter(C, Fg, 'replicate', 'full', 'conv');

% clip to image size
removed = max(0, (size(Fg,1)-1) / 2 - 1);
A = A(removed+1:end-removed,removed+1:end-removed);
B = B(removed+1:end-removed,removed+1:end-removed);
C = C(removed+1:end-removed,removed+1:end-removed);

% shi-tomasi corner scores - minimum eigen values
score = ((A + B) - sqrt((A - B) .^ 2 + 4 * C .^ 2)) / 2;

% remove nans
score(isnan(score)) = 0;

end

function loc = stCornerLocations(score)

% find local maxima
maxScore = max(score(:));
threshold = 0.05 * maxScore;
bw = imregionalmax(score, 8);
bw(score < threshold) = 0;
bw = bwmorph(bw, 'shrink', Inf);

% exclude points on the border
bw(1, :) = 0;
bw(end, :) = 0;
bw(:, 1) = 0;
bw(:, end) = 0;

% find locations
idx = find(bw);
loc = zeros([length(idx), 2], 'like', score);
[loc(:, 2), loc(:, 1)] = ind2sub(size(score), idx);

end

%%%%%%%%%%%%%% MATLAB helper functions in detectMinEigenFeatures %%%%%%%%%%%%

% Compute sub-pixel locations using bi-variate quadratic function fitting.
% Reference: http://en.wikipedia.org/wiki/Quadratic_function
function subPixelLoc = subPixelLocation(metric, loc)

loc = reshape(loc', 2, 1, []);

nLocs = size(loc,3);
patch = zeros([3, 3, nLocs], 'like', metric);
x = loc(1,1,:);
y = loc(2,1,:);
xm1 = x-1;
xp1 = x+1;
ym1 = y-1;
yp1 = y+1;
xsubs = [xm1, x, xp1;
xm1, x, xp1;
xm1, x, xp1];
ysubs = [ym1, ym1, ym1;
y, y, y;
yp1, yp1, yp1];
linind = sub2ind(size(metric), ysubs(:), xsubs(:));
patch(:) = metric(linind);

dx2 = ( patch(1,1,:) - 2*patch(1,2,:) +   patch(1,3,:) ...
+ 2*patch(2,1,:) - 4*patch(2,2,:) + 2*patch(2,3,:) ...
+   patch(3,1,:) - 2*patch(3,2,:) +   patch(3,3,:) ) / 8;

dy2 = ( ( patch(1,1,:) + 2*patch(1,2,:) + patch(1,3,:) )...
- 2*( patch(2,1,:) + 2*patch(2,2,:) + patch(2,3,:) )...
+   ( patch(3,1,:) + 2*patch(3,2,:) + patch(3,3,:) )) / 8;

dxy = ( + patch(1,1,:) - patch(1,3,:) ...
- patch(3,1,:) + patch(3,3,:) ) / 4;

dx = ( - patch(1,1,:) - 2*patch(2,1,:) - patch(3,1,:)...
+ patch(1,3,:) + 2*patch(2,3,:) + patch(3,3,:) ) / 8;

dy = ( - patch(1,1,:) - 2*patch(1,2,:) - patch(1,3,:) ...
+ patch(3,1,:) + 2*patch(3,2,:) + patch(3,3,:) ) / 8;

detinv = 1 ./ (dx2.*dy2 - 0.25.*dxy.*dxy);

% Calculate peak position and value
x = -0.5 * (dy2.*dx - 0.5*dxy.*dy) .* detinv; % X-Offset of quadratic peak
y = -0.5 * (dx2.*dy - 0.5*dxy.*dx) .* detinv; % Y-Offset of quadratic peak

% If both offsets are less than 1 pixel, the sub-pixel location is
% considered valid.
isValid = (abs(x) < 1) & (abs(y) < 1);
x(~isValid) = 0;
y(~isValid) = 0;
subPixelLoc = [x; y] + loc;

subPixelLoc = squeeze(subPixelLoc)';

end

% Compute corner metric value at the sub-pixel locations by using
% bilinear interpolation
function values = computeMetric(metric, loc)
sz = size(metric);

x = loc(:, 1);
y = loc(:, 2);
x1 = floor(x);
y1 = floor(y);
% Ensure all points are within image boundaries
x2 = min(x1 + 1, sz(2));
y2 = min(y1 + 1, sz(1));

values = metric(sub2ind(sz,y1,x1)) .* (x2-x) .* (y2-y) ...
+ metric(sub2ind(sz,y1,x2)) .* (x-x1) .* (y2-y) ...
+ metric(sub2ind(sz,y2,x1)) .* (x2-x) .* (y-y1) ...
+ metric(sub2ind(sz,y2,x2)) .* (x-x1) .* (y-y1);

end

