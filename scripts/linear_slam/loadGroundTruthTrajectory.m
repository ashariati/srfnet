function T = loadGroundTruthTrajectory(dataDir, seq)

poseFile = fullfile(dataDir, 'poses', strcat(sprintf('%02d', seq), '.txt'));
T = dlmread(poseFile);
T = T(:, [4, 8, 12]);

end
