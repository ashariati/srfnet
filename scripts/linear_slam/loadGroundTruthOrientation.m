function orientations = loadGroundTruthOrientation(dataDir, seq)

poseFile = fullfile(dataDir, 'poses', strcat(sprintf('%02d', seq), '.txt'));
data = dlmread(poseFile);

transformations = permute(reshape(data', 4, 3, []), [2, 1, 3]);
orientations = transformations(1:3, 1:3, :);

end