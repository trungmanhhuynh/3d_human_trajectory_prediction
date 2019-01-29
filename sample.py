import random
import math
import torch
import argparse
import time
import shutil
from tqdm import tqdm


from torch import nn, optim
from torch.autograd import Variable
from matplotlib import pyplot as plt
import numpy as np

from models.Alahi_Social_LSTM import Alahi_Social_LSTM
#from models.Model_Linear import Model_Linear
from models.Model_LSTM_1L import Model_LSTM_1L
from models.Model_LSTM_Scene_common import Model_LSTM_Scene_common 

from utils.evaluate import *
from utils.data_loader import *
from utils.visualize import *
from config import *
from utils.scene_grids import get_nonlinear_grids, get_common_grids
from sample import *


def sample(model, data_loader, save_model_file, args, validation = False, test = False):

    #Define model with train = False to set drop out = 0 
    net = model(data_loader, args, train  = False)
    if(args.use_cuda): net = net.cuda() 
    optimizer = optim.RMSprop(net.parameters(), lr = args.learning_rate)

    # Load the trained model from save_model_file
    print("SAMPLE: loading best trained model at {}".format(save_model_file))
    state = torch.load(save_model_file)
    net.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])

    # set number of batches for different modes: test/validation
    if(test):num_batches = data_loader.num_test_batches
    else: num_batches = data_loader.num_validation_batches

    # Intialize error metrics
    mse = 0; nde = 0 ; fde = 0 
    mse_batch_cnts = 0 ; nde_batch_cnts = 0; fde_batch_cnts = 0 
    # Process each batch
    for i in tqdm.tqdm(range(0, num_batches)):

        # Load batch training data 
        if(test): batch  = data_loader.next_test_batch(randomUpdate=False)
        else: batch  = data_loader.next_valid_batch(randomUpdate=False)
        net.init_batch_parameters(batch)                                     # Init hidden states for each target in this batch
        optimizer.zero_grad()

        result_pts = []                                                      # Init the results

        # Process observed trajectory, state each target is updated
        # The results[0:obs] = ground_truth[0:obs]
        for t in range(0,args.observe_length):

            # Get batch data at time t and t+1
            cur_frame = {"loc_off": batch["loc_off"][t], "loc_abs": batch["loc_abs"][t], "frame_pids":  batch["frame_pids"][t]}
            xoff_pred, xabs_pred = net.sample(cur_frame)                    # Sample observered frames to update target's states
            result_pts.append(batch["loc_abs"][t])          # Keep the grouth truth locations

        # Predict future trajectory, each target is present in the last observed frame 
        # is used to predict the next #args.predict_length frames
        cur_frame = {"loc_off": batch["loc_off"][args.observe_length-1], 
                     "loc_abs": batch["loc_abs"][args.observe_length-1],
                     "frame_pids": batch["frame_pids"][args.observe_length-1]}

        predicted_pids = batch["frame_pids"][args.observe_length-1]
        for t in range(args.observe_length -1, args.observe_length + args.predict_length):

            xoff_pred, xabs_pred = net.sample(cur_frame)        # Calculate predicted location
            result_pts.append(xabs_pred)     # Save result
            cur_frame["loc_off"] = xoff_pred                    # Use predicted location as inputs in next frame.
            cur_frame["loc_abs"] = xabs_pred
     
        # calculate mse of this batch 
        mse_valid, batch_mse, numPeds = calculate_mean_square_error(batch, result_pts, predicted_pids, args)
        if(mse_valid):
            mse = mse + batch_mse
            mse_batch_cnts = mse_batch_cnts + numPeds
        # calculate nde of this batch
        nde_valid, batch_nde, numPeds = calculate_mean_square_error_nonlinear(batch, result_pts, predicted_pids, args)
        if(nde_valid):
            nde = nde + batch_nde
            nde_batch_cnts = nde_batch_cnts + numPeds
        # calculate fde of this batch 
        fde_valid, batch_fde, numPeds = calculate_final_displacement_error(batch, result_pts, predicted_pids, args)
        if(fde_valid):
            fde = fde + batch_fde
            fde_batch_cnts = fde_batch_cnts + numPeds
     
        # Save trajectories 
        # plot_trajectory_pts_on_images(batch, i, result_pts, predicted_pids, args)

    # Final mse, nde , fde 
    mse = mse/mse_batch_cnts
    if nde_batch_cnts != 0:
        nde = nde/nde_batch_cnts
    fde = fde/fde_batch_cnts

    return mse, nde , fde

