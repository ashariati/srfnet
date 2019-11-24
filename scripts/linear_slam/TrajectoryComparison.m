clear all;
close all;

% dataDir = '~/Research/Data/srfnet/dataset';
dataDir = 'C:\Users\armon\Documents\Research\Data\srfnet\kitti\dataset';
% seq = 5; % most interesting, 2761 points
seq = 7; % less intersting, 1101 points
% seq = 9; % less interesting 1591 points
translationsFile = 'C:\Users\armon\Documents\Research\Data\srfnet\translations.txt';

groundTruth = loadGroundTruthTrajectory(dataDir, seq);
numPoints = size(groundTruth, 1);
fprintf('Trajectory Length = %d\n', numPoints);

translations = loadTranslationsArray(translationsFile);
rotations = loadGroundTruthOrientation(dataDir, seq);
t = rotateTranslationWindow(rotations, translations, 4);
t = [0; 0; 0; t];

A = linearOdometryModel(numPoints, 4, 1);
x = A \ t;
estimated = reshape(x, 3, [])';

figure;
hold on;
plot(groundTruth(:, 3), groundTruth(:, 1), '-r.');
plot(estimated(:, 3), estimated(:, 1), '-b.');
axis equal;
