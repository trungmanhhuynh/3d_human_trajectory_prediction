
UCY_UNIV sequence.

-------------------------------------------------
INFO: Annotation files given in 
INFO: Video frame rate: 25fps (1 frame = 0.04s). 
INFO: Frame width = 720 , frame height = 576.
---------------------------------------------------------------------
Important files: 

obsmat.txt:  				The file given by BIWI dataset (http://www.vision.ee.ethz.ch/en/datasets/)
(meters)     				Format [frame_number pedestrian_ID pos_x pos_z pos_y v_x v_z v_y ]


original_data_meters:			Extracted data from obsmat.txt 
(meters)  				Results by running get_original_data_meters.m with input is obsmat.txt
					Format [frame_number pedestrian_ID pos_x pos_y]



interpolated_data_meters		Interpolate data from original_data_meters for each frame
(meters)				Results obtained by running script interpolate_data.m


data_meters_2.5fps			Extract interpolated_data_meters at every 10 frames (2.5fps)		
(meters)				This data is very simmilar to data used in SGAN paper. 
					My data has higher precision.
					Results obtained by running the script get_data_2dot5_fps.m

data_pixels_2.5fps 			Convert location (meters) in data_meters into locations in pixels.
(pixels)				Results obtained by running the script getPixelCoordinates.m 

data_normalized_2.5fps 			Convert location pixels.in data_pixels into locations in normalized ranges [-1,1]
(normalized range [-1,1])				
