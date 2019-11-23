%% Inverse compositional formulation
%%
function [V, p] = lucasKanadeFlow(I0, I, patchSize)

if nargin < 3
    patchSize = 5;
end

% spatial gradients -- will come from super resolution network
[Ix, Iy] = spatialGradients(I0);

% feature tracks
corners = pixelsToTrack(Ix, Iy);
locations = corners.Location;
locations = flip(locations, 2);

% rng(0);
% ids = randperm(size(locations, 1), 25);
% locations = locations(ids, :);
% locations = locations(1:707, :);

fprintf('Features to track = %d\n', size(locations, 1));


% steepest descient images and Hessians
jacobian = @(x, p) translationalWarpJacobian(x, 0);
D = steepestDescentImages(Ix, Iy, jacobian);
H = hessianMatrices(D);

% apply Gaussian weighting and invert Hessians
F = gaussianFilter(patchSize);
% F = ones(patchSize);
Hg = imfilter(H, F, 'replicate', 'same', 'conv');
Gg = invertHessianAtLocation(Hg, locations);

% for each track
V = zeros(size(D));
for i=1:size(locations, 1)

    % load inverse Hessian
    Gi = Gg(:, :, i);

    % load track position
    x = locations(i, :);

    fprintf('Estimating flow %d / %d\r', i, size(locations, 1));

    % until convergence
    p = [0, 0];
    dp = Inf;
    iters = 0;
    while norm(dp) > 0.01 && iters < 30;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % IN PRACTICE : 
        % 1) Downsample displacment field
        % 2) Warp coarse w(I_c) = I_w_c
        % 3) Recompute coarse I_t_c = I_w_c - I_0_c
        % 4) SRCNN -> high res I_t

        % warped image
        Iw.data = imwarp(I.data, displacementField(p, size(I.data)));
        Iw.undefMask = I.undefMask;

        % error image -- from super resolution network
        It = temporalGradient(I0, Iw);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % rhs
        R = imfilter(D .* It, F, 'replicate', 'same', 'conv');
        rhs = R(x(1), x(2), :);
        rhs = rhs(:);

        % compute update
        dp = (Gi * rhs)';

        % inverse composition
        p = invCompTranslation(p, dp);

        iters = iters + 1;

    end

    % save flow
    V(x(1), x(2), :) = p;

end

flow = opticalFlow(V(:, :, 1), V(:, :, 2));

imshow(I.data);
hold on;
plot(flow);

keyboard;

end

function G = invertHessianAtLocation(H, locations)

dim = size(H);
k = sqrt(dim(3));

G = zeros(k, k, size(locations, 1), 'like', H);
for i=1:size(locations, 1)

    x = locations(i, :);
    hi = H(x(1), x(2), :);
    Hi = reshape(hi(:), k, k);
    Gi = inv(Hi);
    G(:, :, i) = Gi;

end

end

function H = hessianMatrices(D)

dim = size(D);

% reference indices
[X, Y] = meshgrid(1:dim(2), (1:dim(1)));
locations = [Y(:), X(:)];

H = zeros(dim(1), dim(2), dim(3) * dim(3), 'like', D);
for i=1:size(locations, 1)

    x = locations(i, :);
    hi = reshape(D(x(1), x(2), :), 1, dim(3));
    Hi = hi' * hi;

    H(x(1), x(2), :) = Hi(:);

end

end

function D = steepestDescentImages(Ix, Iy, jacobian)

% jacobian matrices
[X, Y] = meshgrid(1:size(Ix, 2), (1:size(Iy, 1)));
locations = [Y(:), X(:)];

% jacobian
J = jacobian([0, 0]);
Jx = zeros(size(Ix, 1), size(Ix, 2), size(J, 2), 'like', Ix);
Jy = zeros(size(Iy, 1), size(Iy, 2), size(J, 2), 'like', Iy);
for i=1:size(locations, 1)

    % jacobian
    x = locations(i, :);
    J = jacobian(x);

    % save
    Jx(x(1), x(2), :) = J(1, :);
    Jy(x(1), x(2), :) = J(2, :);

end

D = (Ix .* Jx) + (Iy .* Jy);

end

function q = invCompTranslation(p, dp)

q = -dp + p;

end

function D = displacementField(t, dim)

Dx = repmat(t(1), dim(1), dim(2));
Dy = repmat(t(2), dim(1), dim(2));
D = cat(3, Dx, Dy);

end

function y = translationalWarp(x, p)

y = x + p;

end

function J = translationalWarpJacobian(x, p)

J = eye(2);

end

function J = affineWarpJacobian(x, p)

J = [x(1), 0, x(2), 0, 1, 0;
     0, x(1), 0, x(2), 0, 1];

end

function J = projectiveTranslationOnlyWarpJacobian(x, p, K, Z)

fx = K(1, 1);
fy = K(2, 2);
cx = K(1, 3);
cy = K(2, 3);

tx = p(1);
ty = p(2);
tz = p(3);

J = [fx / (Z + tz), 0, (cx*Z - x(1)*Z - fx*tx) / (Z + tz)^2;
     0, fy / (Z + tz), (cy*Z - x(2)*Z - fy*ty) / (Z + tz)^2];

end

function F = gaussianFilter(k)

% gaussian weighting
F = fspecial('gaussian', k, k/3);

end
