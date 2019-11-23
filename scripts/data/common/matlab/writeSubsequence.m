function writeSubsequence(images, subDir)

% extract translations
P = [images.pose]';
T = P(4:4:end, 1:3);

% write translations
tfile = fullfile(subDir, 'translations.txt');
dlmwrite(tfile, T, ' ');

% write images
gifFile = fullfile(subDir, 'sequence.gif');
for i=1:numel(images)

    % image alone
    imageFile = fullfile(subDir, images(i).name);
    imwrite(images(i).data, imageFile, 'png');

    % gif
    % [A, map] = rgb2ind(images(i).data, 256);
    % if i==1
    %     imwrite(A,map,gifFile,'gif','LoopCount',Inf,'DelayTime',0.1);
    % else
    %     imwrite(A,map,gifFile,'gif','WriteMode','append','DelayTime',0.1);
    % end
    
end

end
