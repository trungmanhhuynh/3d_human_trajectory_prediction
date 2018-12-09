import random
import math
import torch
import argparse
import time
import tqdm
import shutil

from torch import nn, optim
from torch.autograd import Variable
from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import inv

def normalized_pixel_to_meters(inputLoc, batch):

	if(batch["dataset_id"] == 0): 
		img_dir ,width, height = './imgs/eth_hotel/', 720, 576
		H = np.array([[ 1.1048200e-02, 6.6958900e-04,-3.3295300e+00],
 					  [-1.5966000e-03, 1.1632400e-02,-5.3951400e+00],
  			 		  [ 1.1190700e-04, 1.3617400e-05, 5.4276600e-01 ]] )
	elif(batch["dataset_id"] == 1):
		img_dir ,width, height = './imgs/eth_univ/', 640, 480
		H = np.array([[2.8128700e-02, 2.0091900e-03, -4.6693600e+00],
					  [8.0625700e-04, 2.5195500e-02, -5.0608800e+00],
					  [3.4555400e-04, 9.2512200e-05,  4.6255300e-01]] )
															
	elif(batch["dataset_id"] == 2):
		img_dir ,width, height = './imgs/ucy_univ/', 720, 576
		H = np.array([[-2.3002776e-02, 5.3741914e-04, 8.6657256e+00],
			  		  [-5.2753792e-04, 1.9565153e-02,-6.0889188e+00],
			          [ 0.0000000e+00,-0.0000000e+00, 1.0000000e+00]])

	elif(batch["dataset_id"] == 3):
		img_dir ,width, height = './imgs/ucy_zara01/', 720, 576
		H = np.array([[-2.5956517e-02, -5.1572804e-18, 7.8388681e+00],
		  		  [-1.0953874e-03, 2.1664330e-02, -1.0032272e+01],
		          [1.9540125e-20,  4.2171410e-19,   1.0000000e+00]])

	elif(batch["dataset_id"] == 4):
		img_dir ,width, height = './imgs/ucy_zara02/', 720, 576
		H = np.array([[-2.5956517e-02,-5.1572804e-18, 7.8388681e+00],
		  		      [-1.0953874e-03, 2.1664330e-02,-1.0032272e+01],
		              [ 1.9540125e-20, 4.2171410e-19, 1.0000000e+00]])
	else: 
		print("Invalid dataset id")
		sys.exit(0) 

	#print("H =" , H)
	# COnvert normalized value o real pixel value
	inputLoc[:,0] = width*(inputLoc[:,0] + 1)/2
	inputLoc[:,1] = height*(inputLoc[:,1] + 1)/2

	# Convert pixel to real-world location
	oneVec = np.ones((inputLoc.shape[0],1))
	tempLoc = np.concatenate((inputLoc,oneVec), axis = 1) # N x 3
	P = np.matmul(H,np.transpose(tempLoc))
	P = np.transpose(P)

	P[:,0] = np.divide(P[:,0],P[:,2])
	P[:,1] = np.divide(P[:,1],P[:,2])

	#print("inputLoc=", inputLoc)
	#print("P =", P[:,0:2])
	#input("here")

	return P[:,0:2]


# Function to calculate MSE for each batch
def calculate_mean_square_error(batch, results, predicted_pids, args):

	selectedPeds = [] 
	# Only care about peds that has obverve length = args.obsever_length 
	for pedId in predicted_pids:
		isSelected = True 
		for t in range(0,args.observe_length):
			presentIdx = np.where( batch["ped_ids_frame"][t] == pedId)[0]
			if(presentIdx.size == 0): # if predicted peds does not have enough
				isSelected = False 	  # T_obs + T_predict frames
				break 
		if(isSelected):
			selectedPeds.append(pedId)
	selectedPeds = np.asarray(selectedPeds) 

	# If there no peds having enough number of frames to calculate mse, then go to next batch
	# return valid = False
	if(selectedPeds.size ==0):
			return False, 0

	# Process each predicted frame
	mseTotal = 0 	
	validPoints = 0 
	for t in range(args.observe_length, args.observe_length + args.predict_length):

		# Process for each ped 
		mseFrame = 0 
		for pedId in selectedPeds:
			# Find idx of this selected ped in batch and in results 
			idxInBatch = np.where( batch["ped_ids_frame"][t] == pedId)[0]
			idxInPredict = np.where(predicted_pids == pedId)[0]
			
			if(idxInBatch.size == 0 or idxInPredict.size == 0):
				continue 

			# Get predicted location and true location
			predict_loc = results[t][idxInPredict]
			true_loc = batch["batch_data_absolute"][t][idxInBatch]
			
			if(args.meters):
				predict_loc = normalized_pixel_to_meters(predict_loc, batch)
				true_loc = normalized_pixel_to_meters(true_loc, batch)

			# Calculate mse in each frame of a ped
			msePed = np.sqrt((predict_loc[:,0]- true_loc[:,0])**2 +  (predict_loc[:,1]- true_loc[:,1])**2)	

			# Calculate mse of all peds in a frame 
			mseTotal = mseTotal + msePed  
			validPoints = validPoints + 1

	if(validPoints == 0):
		return False, 0

   	#Avarage mean square error
	mse = mseTotal/validPoints

	return True, mse

# Function to calculate MSE for each non-linear batch
def calculate_mean_square_error_nonlinear(batch, results, predicted_pids, args):

	# Only care about peds that has obverve length = args.obsever_length 
	selectedPeds = [] 
	for pedId in predicted_pids:
		isSelected = True 
		for t in range(0,args.observe_length):
			presentIdx = np.where( batch["ped_ids_frame"][t] == pedId)[0]
			if(presentIdx.size == 0): # if predicted peds does not have enough
				isSelected = False 	  # T_obs + T_predict frames
				break 
		if(isSelected):
			selectedPeds.append(pedId)
	selectedPeds = np.asarray(selectedPeds) 

	# If there no peds having enough number of frames to calculate mse, then go to next batch
	# return valid = False
	if(selectedPeds.size ==0):
		return False, 0
	
   # Find Peds with non-linear trajectories 
	def is_nonlinear(traj):
		traj = np.vstack(traj[:])
		non_linear_degree = 0

		y_max = traj[-1,1]
		y_min = traj[0,1]
		y_s   =traj[int(traj.shape[0]/2),1]
		if((y_max-y_min)/2 + y_min == 0):
			non_linear_degree = 0
		else:
			non_linear_degree = abs(((y_max - y_min)/2 + y_min - y_s))

		if(non_linear_degree >= 0.1): 
			return True
		else: 
			return False

	nonLinearPeds = []
	for pid in selectedPeds:
		# Get trajectory of this target
		traj = []
		for ti in range(0,args.observe_length + args.predict_length):
			pid_idx = np.where(batch["ped_ids_frame"][ti] == pid)[0]
			if(pid_idx.size == 0):
				break 
			traj.append(batch["batch_data_absolute"][ti][pid_idx])

		# check if ped's trajectory is non-linear 
		if(is_nonlinear(traj)): 
			nonLinearPeds.append(pid)

	nonLinearPeds = np.asarray(nonLinearPeds) 
	# If there is no non-linear trajectories
	if(nonLinearPeds.size == 0):
		return False, 0 
 
	# Process each predicted frame
	mseTotal = 0 	
	validPoints = 0 
								
	for t in range(args.observe_length,args.observe_length + args.predict_length):

		# Process for each ped 
		mseFrame = 0 
		for pedId in nonLinearPeds:
			# Find idx of this selected ped in batch and in results 
			idxInBatch = np.where( batch["ped_ids_frame"][t] == pedId)[0]
			idxInPredict = np.where(predicted_pids == pedId)[0]
			
			if(idxInBatch.size == 0):
				continue 

			# Get predicted location and true location
			predict_loc = results[t][idxInPredict]
			true_loc = batch["batch_data_absolute"][t][idxInBatch]  
			

			if(args.meters):
				predict_loc = normalized_pixel_to_meters(predict_loc, batch)
				true_loc = normalized_pixel_to_meters(true_loc, batch)

			# Calculate mse in each frame of a ped
			msePed = np.sqrt((predict_loc[:,0]- true_loc[:,0])**2 +  (predict_loc[:,1]- true_loc[:,1])**2)	
			validPoints = validPoints + 1
			# Calculate mse of all peds in a frame 
			mseTotal = mseTotal + msePed  


	if(validPoints == 0):
		return False, 0

   	#Avarage mean square error
	mse = mseTotal/validPoints
	
	return True, mse

def calculate_final_displacement_error(batch, results, predicted_pids, args):

	# Only care about peds that has obverve length = args.obsever_length 
	selectedPeds = [] 
	for pedId in predicted_pids:
		isSelected = True 
		for t in range(0,args.observe_length):
			presentIdx = np.where( batch["ped_ids_frame"][t] == pedId)[0]
			if(presentIdx.size == 0): # if predicted peds does not have enough
				isSelected = False 	  # T_obs + T_predict frames
				break 
		if(isSelected):
			selectedPeds.append(pedId)
	selectedPeds = np.asarray(selectedPeds) 

	# If there no peds having enough number of frames to calculate mse, then go to next batch
	# return valid = False
	if(selectedPeds.size ==0):
		return False, 0
	
	# Process each predicted frame
	mseTotal = 0 									# Initilize mse  = 0 

	# Process for each ped 
	mseFrame = 0 
	for pedId in selectedPeds:
		# Find idx of this selected ped in batch and in results 
		idxInBatch = np.where( batch["ped_ids_frame"][-2] == pedId)[0]
		idxInPredict = np.where(predicted_pids == pedId)[0]
		
		# Get predicted location and true location
		predict_loc = results[-1][idxInPredict]
		true_loc = batch["batch_data_absolute"][-2][idxInBatch]  
		

		if(args.meters):
			predict_loc = normalized_pixel_to_meters(predict_loc, batch)
			true_loc = normalized_pixel_to_meters(true_loc, batch)

		# Calculate mse in each frame of a ped
		msePed = np.sqrt((predict_loc[:,0]- true_loc[:,0])**2 +  (predict_loc[:,1]- true_loc[:,1])**2)	
		
		# Calculate mse of all peds in a frame 
		mseFrame = mseFrame + msePed  

	# Total mse for this batch
	mseTotal = 	mseTotal + msePed

   	#Avarage mean square error
	mse = mseTotal/selectedPeds.size

	return True, mse


