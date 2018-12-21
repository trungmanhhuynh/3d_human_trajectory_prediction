%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Script to convert world coordinates
% to image coordinates
% by Anirudh Vemula, Aug 8, 2016
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all

H = dlmread('H.txt');

data = dlmread('data_meters_2.5fps.txt');

% pos = [xpos ypos 1]
pos = [data(:,3) data(:,4) ones(size(data,1),1)];

% Assuming H converts world coordinates to image coordinates
pixel_pos_unnormalized = pinv(H) * pos';

% Normalize pixel pos
% Each column contains [u; v] 
pixel_pos_normalized = bsxfun(@rdivide, pixel_pos_unnormalized([1,2],:), ...
                              pixel_pos_unnormalized(3,:));


% Add frame number and pedestrian ID information to the matrix
pixel_pos_frame_ped = [data(:,1)'; data(:,2)';pixel_pos_normalized];

% pixel_pos_frame_ped 3rd row is y, 4th row is x
% Normalize the coordinates
%pixel_pos_frame_ped(3,:) = pixel_pos_frame_ped(3,:) / 480;
%pixel_pos_frame_ped(4,:) = pixel_pos_frame_ped(4,:) / 640;

% pixel_pos_frame_ped 3rd row is x, 4th row is y
temp = pixel_pos_frame_ped(3,:) ; 
pixel_pos_frame_ped(3,:) = pixel_pos_frame_ped(4,:)  ;
pixel_pos_frame_ped(4,:) = temp ;

% Save the positions to a mat file
%save('pixel_pos.mat', 'pixel_pos_frame_ped');
csvwrite('data_pixels_2.5fps.txt', pixel_pos_frame_ped');

