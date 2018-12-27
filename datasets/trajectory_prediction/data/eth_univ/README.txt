
UCY_Univ sequence.

-------------------------------------------------
INFO: Annotation files given in BIWI dataset (http://www.vision.ee.ethz.ch/en/datasets/)
INFO: Video frame rate: 25fps (1 frame = 0.04s). 
INFO: Frame width = 720 , frame height = 576.
---------------------------------------------------------------------
Important files: 

obsmat.txt:                             The file given by BIWI dataset (http://www.vision.ee.ethz.ch/en/datasets/)
(meters)                                Format [frame_number pedestrian_ID pos_x pos_z pos_y v_x v_z v_y ]
                                        Data in this file has origin NOT in the top-right corner of the image.

original_data_meters.txt:               Extracted data from obsmat.txt 
(meters)                                Results by running get_original_data_meters.m with input is obsmat.txt
                                        Format [frame_number pedestrian_ID pos_x pos_y]
                                        Data in this file has origin NOT in the top-right corner of the image.

original_interpolated_data_meters.txt   Interpolate data from original_data_meters for each frame
(meters)                                Results obtained by running script interpolate_data.m
                                        Data in this file has origin NOT in the top-right corner of the image.

my_data_meters                          Convert to world orgin to right-top corner of the image. 
(meters)                                Data in this file has origin in the top-right corner of the image.

data_meters_2.5fps.txt                  Extract my_data_meters at every 10 frames (2.5fps)      
(meters)                                Results obtained by running the script get_data_2dot5_fps.m
                                       
data_pixels_2.5fps                      Convert location (meters) in data_meters into locations in pixels.
(pixels)                                Results obtained by running the script getPixelCoordinates.m 

data_normalized_2.5fps                  Convert location pixels.in data_pixels into locations in normalized ranges [-1,1]
(normalized range [-1,1])               


Note: 
The data used in my research (data_meters_2.5fps.txt) is different from the one used from SGAN paper,
because the SGAN paper kept the origin some where in the middle of the image.
