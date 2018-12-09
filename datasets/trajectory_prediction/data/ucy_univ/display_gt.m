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

inputVideoFile = 'C:\Users\PDSLab-1\Desktop\Manh\Research\Dataset\ucy\univ\students003.avi';
inputVideoObj = VideoReader(inputVideoFile);

gtFile = 'C:\Users\PDSLab-1\Desktop\Manh\Research\Dataset\ucy\univ\my_peds_annotations_interpolated_raw.txt';
gtData = dlmread(gtFile);

% Write to Video Object
v = VideoWriter('seq_hotel_gt.avi');
v.open(); 

    
    % Plot trajectory for each frame in video sequences.
    % Read frame image
    step = 10;
    h = figure('visible', 'on' ) ; hold on ; 
    nFrames = max(gtData(:,1));
    for frameId = 4000:step:nFrames
        inputVideoObj.CurrentTime = frameId/inputVideoObj.FrameRate ;
        frameImg = readFrame(inputVideoObj);
        h = figure(1) ; hold on ; 
        imshow(frameImg);  hold on ;
        frameData = gtData((gtData(:,1) == frameId),:) ; 
        % toFrame = min(size(trajList{tId}.locations,1),frameId-trajList{tId}.startFrame+1]);
        plot(frameData(:,3),frameData(:,4),'ro','linewidth',2);
        
        % Store figure to video
        F = getframe(h) ;
        writeVideo(v,F.cdata) ;
        %clf
        frameId
    end
    
v.close()
end