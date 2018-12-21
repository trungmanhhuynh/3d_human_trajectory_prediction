%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Description: The script to extract (x,y) data from given data obsmat.txt
%  given by BIWI dataset
%  Author: Huynh
%  Date: 12/11/2018
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function get_original_data_meters
    
    % Read obsmat.txt file given by the dataset BIWI,
    % this data is given more than just (x,y), it also 
    % has vx,vy,vz
    data = dlmread('obsmat.txt');
    
    % Extract frameid, targetid, x, y 
    xy_data = [data(:,1), data(:,2), data(:,3), data(:,5)] ;
   
    csvwrite('original_data_meters.txt', xy_data);

    fprintf("done\n")

end 