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
from utils.metric_conversions import *

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
			return False, 0, 0

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
			
			# convert metrics
			predict_loc = convert_input2output_metric(predict_loc, batch["dataset_id"], args.input_metric, args.output_metric)
			true_loc = convert_input2output_metric(true_loc, batch["dataset_id"], args.input_metric, args.output_metric)

			# Calculate mse in each frame of a ped
			msePed = np.sqrt((predict_loc[:,0]- true_loc[:,0])**2 +  (predict_loc[:,1]- true_loc[:,1])**2)	

			# Calculate mse of all peds in a frame 
			mseTotal = mseTotal + msePed  
			validPoints = validPoints + 1

	if(validPoints == 0):
		return False, 0, 0

	return True, mseTotal , validPoints

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
		return False, 0, 0
	
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
		return False, 0 ,0 
 
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
			
			# convert metrics
			predict_loc = convert_input2output_metric(predict_loc, batch["dataset_id"], args.input_metric, args.output_metric)
			true_loc = convert_input2output_metric(true_loc, batch["dataset_id"], args.input_metric, args.output_metric)

			# Calculate mse in each frame of a ped
			msePed = np.sqrt((predict_loc[:,0]- true_loc[:,0])**2 +  (predict_loc[:,1]- true_loc[:,1])**2)	
			validPoints = validPoints + 1
			# Calculate mse of all peds in a frame 
			mseTotal = mseTotal + msePed  


	if(validPoints == 0):
		return False, 0, 0
	
	return True, mseTotal, validPoints

def calculate_final_displacement_error(batch, results, predicted_pids, args):

	# Only care about peds that has obverve length = args.obsever_length 
	# and has at least 1 frame to predict 
	selectedPeds = [] 
	for pedId in predicted_pids:
		isSelected = True 
		for t in range(0,args.observe_length + args.predict_length):
			presentIdx = np.where(batch["ped_ids_frame"][t] == pedId)[0]
			if(presentIdx.size == 0): # if predicted peds does not have enough
				isSelected = False 	  # T_obs + T_predict frames
				break 
		if(isSelected):
			selectedPeds.append(pedId)
	selectedPeds = np.asarray(selectedPeds) 

	# If there no peds having enough number of frames to calculate mse, then go to next batch
	# return valid = False
	if(selectedPeds.size ==0):
		return False, 0, 0

	'''
	# Find the last frame of each ped_id in selected pedestrians 
	lastFrameList = []
	for pedId in selectedPeds:
		for t in range(args.observe_length , args.observe_length + args.predict_length):
			presentIdx = np.where(batch["ped_ids_frame"][t] == pedId)[0]

			# if ped is not present in this frame t, then it's
			# last frame is previous frame
			if(presentIdx.size == 0): 
				lastFrameList.append(t-1)
				break 

		# if ped is present in the final frame
		if(presentIdx.size is not 0):
			lastFrameList.append(t)

	lastFrameList = np.asarray(lastFrameList) 
	'''

	# Initialize errors
	mseTotal = 0 									# Initilize mse  = 0 
	mseFrame = 0 

	lastFrame = args.observe_length + args.predict_length - 1
	for pedId in selectedPeds:

		# Find idx of this selected ped in batch and in results 
		idxInBatch = np.where(batch["ped_ids_frame"][lastFrame] == pedId)[0]
		idxInPredict = np.where(predicted_pids == pedId)[0]
		
		# Get predicted location and true location
		predict_loc = results[lastFrame][idxInPredict]
		true_loc = batch["batch_data_absolute"][lastFrame][idxInBatch]  
			
		# convert metrics
		predict_loc = convert_input2output_metric(predict_loc, batch["dataset_id"], args.input_metric, args.output_metric)
		true_loc = convert_input2output_metric(true_loc, batch["dataset_id"], args.input_metric, args.output_metric)

		# Calculate mse in each frame of a ped
		msePed = np.sqrt((predict_loc[:,0]- true_loc[:,0])**2 +  (predict_loc[:,1]- true_loc[:,1])**2)	

		# Calculate mse of all peds in a frame 
		mseTotal = mseFrame + msePed

	return True, mseTotal, selectedPeds.size


