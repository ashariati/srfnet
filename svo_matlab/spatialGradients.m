function [Ix, Iy] = spatialGradients(I)

G = single(I.data);
G(I.undefMask) = NaN;

[Ix, Iy] = imgradientxy(G);

% Ix = imfilter(G, [-1 0 1] , 'replicate', 'same', 'conv');
% Iy = imfilter(G, [-1 0 1]', 'replicate', 'same', 'conv');

end
