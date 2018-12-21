%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Description: The script to extract interpolated_data_meters at 10fps 
%
%  Author: Huynh
%  Date: 12/11/2018
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function get_data_2dot5_fps

    %interpolated_data_meters is at 25fps 
    data = dlmread('interpolated_data_meters.txt');
    
    frame_list = unique(data(:,1));
    data_meters_2dot5_fps = [] ;
    for i=1:size(frame_list,1)
        
        frameid = frame_list(i) ;

        % collect data at every 10 frames. 
        if(mod(frameid - 1, 10)== 0)                % because index frame start at 1
            frameid
            % Get data in this frame i 
            frame_data = data(data(:,1) == frameid,:) ;
        
            % Store it
            data_meters_2dot5_fps  = [data_meters_2dot5_fps ; frame_data] ;
        end 
    end
    csvwrite('data_meters_2.5fps.txt', data_meters_2dot5_fps);
    fprintf("done\n")
end 