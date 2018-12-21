% File name:  normalize data.m 
% Description: function to normalize data from pixel values (image
%               coordinate) to range [0,1] or [-1,1]
% How to Use: 
%   - Specify input file name.
%   - run normalize_data('eth_univ', 1) # mode = 0 to normalize to range [0,1]
%                                 # mode = 1 to normalize to range [-1,1] 
%   - Results of normalized data is stored as same folder of input data 
% Author : Huynh Manh
% Last Modified: 09/04/2018


function normalize_data

pixel = 0 ; % = 1 to normarlize pixels to a specified range
            % = 0 to normalize meters range [0, 20] to a specified range
filename = strcat('data_meters_2.5fps.txt');
inputData = dlmread(filename); 
  
if(pixel)
    frameWidth  = 720; 
    frameHeight = 576;  
else
    frameWidth  = 20 ; 
    frameHeight = 20 ;    
end 

normalizedData = inputData ; 
mode = 1 ; % convert to range [-1,1]

if(mode ==0)
    disp("Normalize (x,y) locations into range [0,1]")
    normalizedData(:,3) = normalizedData(:,3)/frameWidth ; % normalize x
    normalizedData(:,4) = normalizedData(:,4)/frameHeight ; % normalize y
    % Write file 
    outfile = strcat('..\data\',dataset,'\',dataset,'_data(0,1).txt');
    csvwrite(outfile, normalizedData);
    disp("Done");
elseif(mode == 1) 
    disp("Normalize (x,y) locations into range [-1,1]")
    normalizedData(:,3) = 2*normalizedData(:,3)/frameWidth -1 ; % normalize x
    normalizedData(:,4) = 2*normalizedData(:,4)/frameHeight -1 ; % normalize y
    % Write file 
    csvwrite('data_normalized_meters_2.5fps.txt', normalizedData);
    disp("Done")
else 
   error('mode can be either 0 or 1'); 

end     

end 