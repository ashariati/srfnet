close all;
clear all;

% parameters
sequences = {'advio-10'};
datasetDir = '~/Projects/CVIO/data/advio/dataset';
scales = [4, 8, 16, 32];

for k=1:length(sequences)

    seqId = sequences{k};

    % load data
    videoDir = fullfile(datasetDir, seqId, 'iphone');
    reader = VideoReader(fullfile(videoDir, 'frames.mov'));

    % for each scale
    for scale=scales

        % initialize new video writer
        writer = VideoWriter(fullfile(videoDir, sprintf('frames_%dx', scale)), 'Grayscale AVI');
        writer.FrameRate = reader.FrameRate;
        open(writer);

        fprintf('Creating simulation %s ... \n', sprintf('frames_%dx.avi', scale));

        % for each frame
        while reader.hasFrame 

            % read frame
            frame = readFrame(reader);

            % coarse simulation
            coarseFrame = imresize(frame, 1 / scale, 'lanczos3');

            % interpolation
            interpFrame = imresize(coarseFrame, scale, 'bicubic');

            % write to file
            % writeVideo(writer, interpFrame);
            writeVideo(writer, rgb2gray(interpFrame)');

        end

        % reset
        reader.CurrentTime = 0;
        close(writer);

    end

end
