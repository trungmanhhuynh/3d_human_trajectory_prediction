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
from models.Model_LSTM_Scene import *

from utils.evaluate import *
from utils.data_loader import *
from utils.visualize import *
from config import *
from grid import get_allow_grids
from sample import *


def sample(model, data_loader, save_model_file, args, validation = False, test = False):

    #Define model with train = False to set drop out = 0 
    net = model(args, train  = False)
    if(args.use_cuda):
        net = net.cuda() 
    optimizer = optim.RMSprop(net.parameters(), lr = args.learning_rate)

    # Load the trained model from save_model_file
    print("SAMPLE: loading best trained model at {}".format(save_model_file))
    state = torch.load(save_model_file)
    net.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
   
    # Initialize scene states for all dataset.
    if(args.use_scene):
        scene_h0_list = state['scene_h0_list'].clone()
        scene_c0_list = state['scene_c0_list'].clone()    
        allow_grid_list = get_allow_grids(data_loader,args)
        net.allow_grid_list = allow_grid_list

    # set number of batches for different modes: test/validation
    if(test):
        num_batches = data_loader.num_test_batches
    else: # validation 
        num_batches = data_loader.num_validation_batches


    # Intialize error metrics
    mse = 0; nde = 0 ; fde = 0 
    mse_batch_cnts = 0 ; nde_batch_cnts = 0; fde_batch_cnts = 0 
    # Process each batch
    for i in tqdm.tqdm(range(0, num_batches)):

        # Load batch training data 
        if(test):
            batch  = data_loader.next_test_batch(randomUpdate=False)
        else: 
            batch  = data_loader.next_valid_batch(randomUpdate=False)

        # Set hidden states for each target in this batch
        nPeds = batch["ped_ids"].size
        h0 = Variable(torch.zeros(args.num_layers, nPeds, args.rnn_size))
        c0 = Variable(torch.zeros(args.num_layers, nPeds, args.rnn_size))
        if(args.use_cuda): 
            h0,c0 = h0.cuda(), c0.cuda()      
          
        # Clear things before forwarding a batch
        optimizer.zero_grad()

        # init the results
        result_pts = [] 
        result_ids = [] 

        # Process observed trajectory, state each target is updated
        # The results[0:obs] = ground_truth[0:obs]
        for t in range(0,args.observe_length):

            # Get input data of time t 
            xoff, xoff_next = batch["batch_data_offset"][t] , batch["batch_data_offset"][t+1]
            xabs, xabs_next = batch["batch_data_absolute"][t],  batch["batch_data_absolute"][t+1]
            xoff = Variable(torch.from_numpy(xoff)).float()              # x ~ [1,batch_size,2]
            xoff_next = Variable(torch.from_numpy(xoff_next)).float()
            xabs = Variable(torch.from_numpy(xabs)).float()             
            xabs_next = Variable(torch.from_numpy(xabs_next)).float()              
            if(args.use_cuda): 
                xoff, xoff_next  = xoff.cuda(), xoff_next.cuda()
                xabs, xabs_next  = xabs.cuda(), xabs_next.cuda()

            # Get/Set the hidden states of targets in current frames
            indices = np.where(batch["ped_ids_frame"][t][:, None] ==  batch["ped_ids"][None, :])[1]            
            indices = Variable(torch.LongTensor(indices))
            if(args.use_cuda): 
                indices = indices.cuda()
            h0_t =  torch.index_select(h0,1,indices)
            c0_t =  torch.index_select(c0,1,indices)
            net.set_target_hidden_states(h0_t, c0_t)                # Set hidden states for this model

            # Get the hidden states of scene in current frame
            #scene_grid_h0, scene_grid_c0 ~ [num_layers,nPeds,rnn_size]
            if(args.use_scene):
                scene_grid_h0, scene_grid_c0, list_of_grid_id = get_scene_states(xabs, batch["dataset_id"], \
                                         args.scene_grid_num, scene_h0_list, scene_c0_list, args)
                net.set_scene_hidden_states(scene_grid_h0, scene_grid_c0)                # Set hidden states for this model
                net.current_grid_list = list_of_grid_id 
                net.current_dataset = batch["dataset_id"] 


            # set others model parameters  
            net.batch_size = batch["ped_ids_frame"][t].size
            # Forward pass 
            mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits = net.forward(xoff, xabs)

            # Store target's hidden states back to storage
            h0[:,indices,:] = net.i_h0.clone()
            c0[:,indices,:] = net.i_c0.clone()

            # TO-DO: Update scene's states
            if(args.use_scene):
                grid_id_unique = np.unique(list_of_grid_id.data.cpu().numpy())
                for grid_id in grid_id_unique:
                    if(grid_id in allow_grid_list[batch["dataset_id"]]):
                        for b in range(net.batch_size):
                            if(list_of_grid_id[b].item() == grid_id and random.randint(0, 1) == 1):
                                scene_c0_list[batch["dataset_id"],grid_id] =  net.scene_grid_c0[:,b,:].data.clone() 
            # save the results.
            result_pts.append(batch["batch_data_absolute"][t])
            #result_ids.append(batch["ped_ids_frame"][t])


        #print("result_pts=", result_pts)
        #print("result_ids=", result_ids)
        #input("here")

        #-----------------------------------------------------------------
        # Generate predicted trajectories
        #-----------------------------------------------------------------
        # Predict future trajectory, each target is present in the last observed frame 
        # is used to predict the next #args.predict_length frames
        predicted_pids = batch["ped_ids_frame"][args.observe_length-1]

        # Get input data of time  args.observe_length-1
        xoff = batch["batch_data_offset"][args.observe_length-1] 
        xabs = batch["batch_data_absolute"][args.observe_length-1]
        xoff = Variable(torch.from_numpy(xoff)).float()              # x ~ [1,batch_size,2]
        xabs = Variable(torch.from_numpy(xabs)).float()             
        if(args.use_cuda): 
            xoff  = xoff.cuda()
            xabs  = xabs.cuda()

        # Get/Set the hidden states of targets at args.observe_length-1
        indices = np.where(predicted_pids[:,None] ==  batch["ped_ids"][None,:])[1]
        indices = Variable(torch.LongTensor(indices))
        if(args.use_cuda): 
            indices = indices.cuda()

        for t in range(args.observe_length -1, args.observe_length + args.predict_length):

            h0_t =  torch.index_select(h0,1,indices)
            c0_t =  torch.index_select(c0,1,indices)
            net.set_target_hidden_states(h0_t, c0_t)                # Set hidden states for this model

            # Get the hidden states of scene in current frame
            #scene_grid_h0, scene_grid_c0 ~ [num_layers,nPeds,rnn_size]
            if(args.use_scene):
                scene_grid_h0, scene_grid_c0, list_of_grid_id = get_scene_states(xabs, batch["dataset_id"], \
                                         args.scene_grid_num, scene_h0_list, scene_c0_list, args)
                net.set_scene_hidden_states(scene_grid_h0, scene_grid_c0)                # Set hidden states for this model
                net.current_grid_list = list_of_grid_id 
                net.current_dataset = batch["dataset_id"] 


            # set others model parameters  
            net.batch_size = len(predicted_pids)
            
            # Forward pass 
            mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits = net.forward(xoff, xabs)

            # Store target's hidden states back to storage
            h0[:,indices,:] = net.i_h0.clone()
            c0[:,indices,:] = net.i_c0.clone()

            #Re-calculate inputs for next prediction step
            xoff = torch.cat([mu1.data,mu2.data], dim = 1)
            xabs = xabs + xoff

            # save the results.
            result_pts.append(xabs.cpu().data.numpy())

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
        #plot_trajectory_pts_on_images(batch, i, result_pts, predicted_pids, args)



    # Final mse, nde , fde 
    print("mse =", mse)
    print("mse_batch_cnt", mse_batch_cnts)

    print("fde =", fde)
    print("fde_batch_cnts", fde_batch_cnts)
    mse = mse/mse_batch_cnts
    nde = nde/nde_batch_cnts
    fde = fde/fde_batch_cnts

    return mse, nde , fde

