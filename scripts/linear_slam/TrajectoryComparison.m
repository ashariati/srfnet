clear all;
close all;

% dataDir = '~/Research/Data/srfnet/dataset';
dataDir = 'C:\Users\armon\Documents\Research\Data\srfnet\kitti\dataset';
seq = 5; % most interesting, 2761 points
% seq = 7; % less intersting, 1101 points
% seq = 9; % less interesting 1591 points
translationsFile = 'C:\Users\armon\Documents\Research\Data\srfnet\kitti_derot_results\05\translations.txt'

groundTruth = loadGroundTruthTrajectory(dataDir, seq);
numPoints = size(groundTruth, 1);
fprintf('Trajectory Length = %d\n', numPoints);

totalDistance = 0;
for i=1:numPoints-1
    totalDistance = totalDistance + norm(groundTruth(i+1, :) - groundTruth(i, :)); 
end

translations = loadTranslationsArray(translationsFile);
rotations = loadGroundTruthOrientation(dataDir, seq);
t = rotateTranslationWindow(rotations, translations, 4);
t = [0; 0; 0; t];

A = linearOdometryModel(numPoints, 4, 1);
x = A \ t;
estimated = reshape(x, 3, [])';

endpointError = norm(estimated(end, :) - groundTruth(end, :));

figure;
title({sprintf('Estimated Trajectory for Sequence %d', seq);
    sprintf('Trajectory Length = %.2f (m), Error = %.2f (m)', totalDistance, endpointError)});
hold on;
plot(groundTruth(:, 3), groundTruth(:, 1), '-r');
plot(estimated(:, 3), estimated(:, 1), '-b');
legend('Ground Truth', 'Estimated');
axis equal;
