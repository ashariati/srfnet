close all;
clear all;

% dataset
dataset = 'kitti';
dataDir = fullfile('~/Projects/CVIO/data', dataset, 'coarsesim');

% sequence
sequence = '00';
subsequence = '00336';
scale = 4;
subseqDir = fullfile(dataDir, sequence, 'subsequences', sprintf('%dx', scale), subsequence);

% image sequence
images = loadImagesFromDir(subseqDir);
