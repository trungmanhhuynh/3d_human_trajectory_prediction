function display_gt
%This function show locations of all target in frames
%Input:
%  + File(*.txt) has each col as follows:
%       1. FrameID
%       2. Pedestrian ID
%       3. xcenter of bounding box
%       4. ycenter of bounding box
%   + video squence
%
%Output:
%   + Display peds location in a given frame
%
close all ;
clear all; 

inputVideoFile = 'C:\Users\PDSLab-1\Desktop\Manh\Research\Dataset\eth\univ\seq_eth.avi';
inputVideoObj = VideoReader(inputVideoFile);
nVideoFrames = round(inputVideoObj.Duration*inputVideoObj.FrameRate) ;

gtFile = 'C:\Users\PDSLab-1\Desktop\Manh\Research\Dataset\eth\univ\my_peds_annotations_interpolated_raw.txt';
gtData = dlmread(gtFile);


    
    % Plot trajectory for each frame in video sequences.
    % Read frame image
    step = 1 ;
    for frameId = 9000:step:size(gtData,1)
        inputVideoObj.CurrentTime = frameId/inputVideoObj.FrameRate ;
        frameImg = readFrame(inputVideoObj);
        figure(1), imshow(frameImg); hold on ; 
        frameData = gtData((gtData(:,1) == frameId),:) ; 
        % toFrame = min(size(trajList{tId}.locations,1),frameId-trajList{tId}.startFrame+1]);
        plot(frameData(:,3),frameData(:,4),'ro','linewidth',2);
        frameId
    end
    

end