clear all;
close all;

dataDir = '~/Research/Data/srfnet/dataset';
% seq = 5; % most interesting, 2761 points
seq = 7; % less intersting, 1101 points
% seq = 9; % less interesting 1591 points

groundTruth = loadGroundTruthTrajectory(dataDir, seq);
fprintf('Trajectory Length = %d\n', size(groundTruth, 1));

plot(groundTruth(:, 3), groundTruth(:, 1), '-r.');
axis equal
