# File name: estimateHomographyMatrix.py
# Description: Estimate the homography matrix to convert 
# pixels (camera coordinates) to meter (world coordinate)
# in CROWDS dataset. 
# For other datasets, the values of each estimated locations 
# must be changed. 
# Author: Huynh Manh
# Date: 12/19/2018

import cv2
import numpy as np 


# Crowds dataset info
img_width = 720 
img_height = 576

# find the homography matrix to convert from camera coordinates (pixels)
# to world coordinates (meters). The world coordinate has origin
# at the top-left corner of the car. 

# 4  corners of the car in pixels
pts_img = np.array([[476, 117], [562, 117], [562, 311],[476, 311]])

# 4 corners of the car in meters 
pts_wrd = np.array([[0, 0], [1.81, 0], [1.81, 4.63],[0, 4.63]])

# Find homography matrix 
h1, status = cv2.findHomography(pts_img, pts_wrd)

#print(h)

# Move the origin at top-left corner of the car to bottom-left corner of the location
# corresponding to the bottom-left corner of the image. 

# pixel location of bottom left image 
# this index is litte wrong, but it produces the same data results 
# used in SGAN papers.
bl_pix = np.array([0,576,1])

# world-coordinate of bottom-left 
bl_meter = h1.dot(bl_pix)

# homography matrix to move origins in world-coordinate,
# it consists of rotation and translation matrix.
h2 = np.array([[1, 0, abs(bl_meter[0])],[0, -1 , abs(bl_meter[1])],[0, 0, 1]])

# the final homography matrix is:
h = h2.dot(h1)

print("the transformation matrix is:")
print(h)

print("testing pixel (1,576,1), it should be [0,0,1] in world-coordinate"); 
test_pt1_pix = np.array([0,576,1])
test_pt1_met = h.dot(test_pt1_pix)

print("test_pt1_pix =", test_pt1_pix)
print("test_pt1_met =", test_pt1_met)

print("testing pixel (206,268,1), it should be [4.33558125829, 7.35072183117,1] in world-coordinate \
       same data used in SGAN (data from zara_01, frame 1, human id 8"); 
test_pt2_pix = np.array([206,268,1])
test_pt2_met = h.dot(test_pt2_pix)

print("test_pt2_pix =", test_pt2_pix)
print("test_pt2_met =", test_pt2_met)
