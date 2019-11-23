clear all;

% add common functions
addpath('../../common/matlab');

% parameters
sequences = {'V1_01_easy', 'V1_02_medium', 'V2_01_easy', 'V2_02_medium'};
simDir = '~/Projects/CVIO/data/euroc/coarsesim';
dataDir = '~/Projects/CVIO/data/euroc/dataset';
windowSize = 20;
stride = 19;
scales = [2, 4, 8, 16, 32];

for k = 1:length(sequences)

    seqId = sequences{k};

    destDir = fullfile(simDir, seqId, 'subsequences');
    seqDir = fullfile(dataDir, 'sequences', seqId);
    imgDir = fullfile(dataDir, 'sequences', seqId, 'images');
    poseDir = fullfile(dataDir, 'poses');
    
    % initialize directories
    mkdir(simDir);
    mkdir(destDir);
    
    % load calibration
    calibFile = fullfile(seqDir, 'calib.txt');
    K = importdata(calibFile, ' ');
    
    % load ground truth pose data
    poseFile = fullfile(poseDir, strcat(seqId, '.txt'));
    poseData = dlmread(poseFile, ' ');
    poseTime = poseData(:, 1);
    poses = permute(reshape(poseData(:, 2:end)', 4, 4, []), [2, 1, 3]);

    % load image capture times
    imageTimeFile = fullfile(seqDir, 'times.txt');
    imageTime = dlmread(imageTimeFile, ' ');

    % interpolate poses at image capture times
    poses = interpolatePoses(poses, poseTime, imageTime);
    
    % read all images
    nImages = numel(dir(strcat(imgDir, '/*.png')));
    for i=1:nImages
        imageName = strcat(sprintf('%05d', i-1), '.png');
        imageFile = fullfile(imgDir, imageName);
        images(i).name = imageName;
        images(i).data = repmat(imread(imageFile), 1, 1, 3);
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
    
            fprintf('Writing subsequence %d / %d ... \r', j, size(subseqIds, 2));
    
        end
        fprintf('\n');
    
    end

end
    
