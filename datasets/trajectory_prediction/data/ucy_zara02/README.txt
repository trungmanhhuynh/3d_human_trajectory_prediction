
UCY_ZARA02 sequence.

-------------------------------------------------
INFO:
INFO: Video frame rate: 25fps (1 frame = 0.04s). 
INFO: Frame width = 720 , frame height = 576.
---------------------------------------------------------------------
Important files: 

crowds_zara02.vsp  			The file given by crowds dataset (https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data)
(pixels)     				Control points (where target turns) of each target


original_data_pixels			Extracted data from ni_examples.vsp
(pixels)				Results by running XXX with input is ni_examples.vsp
					Format [frame_number pedestrian_ID pos_x pos_y]


interpolated_data_pixels		Interpolate data from original_data_pixelsfor each frame
(pixels)				Results obtained by running script interpolate_data.m


data_pixels_2.5fps 			Extract interpolated_data_meters at every 10 frames (2.5fps)
(pixels)				Results obtained by running the script get_data_2dot5_fps.m


data_normalized_2.5fps 	  Convert location pixels.in data_pixels into locations in normalized ranges [-1,1]
(normalized range [-1,1])		


data_meters_2.5fps					
(meters)				This data is very simmilar to data used in SGAN paper.(
					SGAN data start frame index with 0 using interpolation at frame 0
					so it is less accurate than using gt data with frame index 1.
					My data has higher precision.
					

		
