function derotatedImages = derotateSequence(images, K)

% align poses
G0 = images(1).pose;
derotatedImages(1).pose = eye(4);
for i=2:numel(images)
    derotatedImages(i).pose = inv(G0) * images(i).pose;
end

% apply homogrophy
derotatedImages(1).data = images(1).data;
derotatedImages(1).name = images(1).name;
for i=2:numel(images)
    R = derotatedImages(i).pose(1:3, 1:3);
    H = K * R' * inv(K);
    derotatedImages(i).data = warpImage(images(i).data, H);
    derotatedImages(i).name = images(i).name;
end
