function convert2meters


clear all

H = dlmread('H.txt');

data = dlmread('data_pixels_2.5fps.txt');

% pos = [xpos ypos 1]
pos = [data(:,3) data(:,4) ones(size(data,1),1)];

% Assuming H converts world coordinates to image coordinates
meter_pos = H * pos';

% Normalize pixel pos
% Each column contains [u; v] 
meter_pos = bsxfun(@rdivide, meter_pos([1,2],:), ...
                              meter_pos(3,:));

meter_pos = meter_pos';

meter_pos = [data(:,[1 2]) ,meter_pos];
csvwrite('data_meters_2.5fps.txt', meter_pos);
fprintf("done\n")

end 