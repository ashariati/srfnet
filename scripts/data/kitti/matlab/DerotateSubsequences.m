clear all;

% add common functions
addpath('../../common/matlab');

% parameters
sequences = {'00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10'};
simDir = '~/Projects/CVIO/data/kitti/coarsesim';
dataDir = '~/Projects/CVIO/data/kitti/dataset';
windowSize = 4;
stride = 2;
scales = [1, 4, 8, 16];

for k = 1:length(sequences)

    seqNum = sequences{k};

    destDir = fullfile(simDir, seqNum, 'subsequences');
    seqDir = fullfile(dataDir, 'sequences', seqNum);
    imgDir = fullfile(dataDir, 'sequences', seqNum, 'image_0');
    poseDir = fullfile(dataDir, 'poses');
    
    % initialize directories
    mkdir(simDir);
    mkdir(destDir);
    
    % load calibration
    calibFile = fullfile(seqDir, 'calib.txt');
    calib = importdata(calibFile, ' ');
    K = reshape(calib.data(1, :), 4, 3)';
    K = K(1:3, 1:3);
    
    % load poses
    poseFile = fullfile(poseDir, strcat(seqNum, '.txt'));
    poseData = dlmread(poseFile, ' ');
    poseData = [poseData, repmat([0, 0, 0, 1], size(poseData, 1), 1)];
    poses = permute(reshape(poseData', 4, 4, []), [2, 1, 3]);
    
    % read all images
    nImages = numel(dir(strcat(imgDir, '/*.png')));
    for i=1:nImages
        imageName = strcat(sprintf('%06d', i-1), '.png');
        imageFile = fullfile(imgDir, imageName);
        images(i).name = imageName;

        imageRaw = imread(imageFile);
        imageCrop = imageRaw(5:end-4, 5:end-5); % Center KITTI Crop
        images(i).data = repmat(imageCrop, 1, 1, 3);

        images(i).pose = poses(:, :, i);
        fprintf('Reading images ... %%%3.0f\r', 100 * (i / nImages));
    end
    fprintf('\n');
    
    % subsequence indices
    imgIds = 0:nImages-1;
    startIds = imgIds(1:stride:end);
    startIds((startIds + windowSize) > nImages) = [];
    subseqIds = repmat(startIds, windowSize, 1) + (0:windowSize-1)';
    
    % for each downsample scale
    for s=scales
        
        fprintf('Writing subsequences at 1/%d resolution ... \n', s);
    
        % initialize directory for scale
        scaleDir = fullfile(destDir, strcat(num2str(s), 'x'));
        mkdir(scaleDir);
    
        % for each subsequence
        for j=1:size(subseqIds, 2)
    
            % initialize subdirectory for sequence
            subDir = fullfile(scaleDir, sprintf('%05d', j-1));
            mkdir(subDir);
    
            % ids
            ids = subseqIds(:, j);
    
            % downsample
            [downsampledImages, Kd] = downsampleSequence(images(ids+1), K, s);
    
            % derotate
            derotatedImages = derotateSequence(downsampledImages, Kd);
    
            % write out
            writeSubsequence(derotatedImages, subDir);

            % write calibration
            scaledCalibFile = fullfile(subDir, 'scaled_calibration.txt');
            dlmwrite(scaledCalibFile, Kd, ' ');
    
            fprintf('Writing subsequence %d / %d ... \r', j, size(subseqIds, 2));
    
        end
        fprintf('\n');
    
    end

end
    
