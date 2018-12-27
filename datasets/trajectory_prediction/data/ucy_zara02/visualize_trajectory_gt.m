function visualize_trajectory_gt
%This function visualize trajectory of targets
%Input:
%  + File(*.txt) has each col as follows:
%       1. FrameID
%       2. Pedestrian ID
%       3. xcenter of bounding box
%       4. ycenter of bounding box
%   + video squence
%
%Output:
%   + Display trajectory in a given frame
%
close all ;
clear all; 

% Read video file
inputVideoFile = 'D:\CUDenver\Research\human_tracjectory_prediction\videos\ucy_zara02\ucy_zara02.avi';
inputVideoObj = VideoReader(inputVideoFile);

% Read ground truth file
gtFile = 'D:\CUDenver\Research\human_tracjectory_prediction\data\ucy_zara02\data_pixels_2.5fps.txt';
gtData = dlmread(gtFile);

% Which frame do you want to plot ?
frameNumber = 9971 ;

% Plot data (location) on frame image
inputVideoObj.CurrentTime = frameNumber/inputVideoObj.FrameRate ;
frameImg = readFrame(inputVideoObj);
figure(1), imshow(frameImg); hold on ; 
frameData = gtData((gtData(:,1) == frameNumber),:) ; 
plot(frameData(:,3),frameData(:,4),'ro','linewidth',2);

% Plot normalized data ?
figure()
gtNormalizedFile = 'D:\CUDenver\Research\human_tracjectory_prediction\data\ucy_zara02\data_normalized_meters_2.5fps.txt';
gtNormalized = dlmread(gtNormalizedFile);

NormalizedframeData = gtNormalized((gtNormalized(:,1) == frameNumber),:) ; 
plot(NormalizedframeData(:,3),NormalizedframeData(:,4),'ro','linewidth',2);
set(gca,'XAxisLocation','top','YAxisLocation','left','ydir','reverse');
axis([-1 1 -1 1])


end



