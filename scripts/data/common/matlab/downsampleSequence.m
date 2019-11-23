function [downsampledImages, Kd] = downsampleSequence(images, K, scale)

% copy intrinsics
Kd = K;

if scale == 1
    downsampledImages = images;
    return;
end

% adjust intrinsics
Kd(1:2, :) = Kd(1:2, :) / scale;

% resize
for i=1:numel(images)
    downsampledImages(i).data = imresize(images(i).data, 1/scale, 'lanczos3');
    downsampledImages(i).name = images(i).name;
    downsampledImages(i).pose = images(i).pose;
end

