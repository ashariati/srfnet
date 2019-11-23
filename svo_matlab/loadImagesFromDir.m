function images = loadImagesFromDir(directory, ext)

if nargin < 2
    ext = 'png';
end

imageFiles = dir(fullfile(directory, sprintf('*.%s', ext)));
[~, sortedOrder] = sort({imageFiles.name});
imageFiles = imageFiles(sortedOrder);
for i=1:numel(imageFiles)

    % read image
    I = imread(fullfile(directory, imageFiles(i).name)); 

    % pixels beyond the fov of reference frame
    undefMask = (I(:, :, 1) == intmin('uint8')) & ...
        (I(:, :, 2) == intmin('uint8')) & ...
        (I(:, :, 3) == intmax('uint8'));

    % save
    images(i).data = I(:, :, 1);
    images(i).undefMask = undefMask;

end

end
