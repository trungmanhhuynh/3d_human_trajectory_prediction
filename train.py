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

from models.Alahi_Social_LSTM import Alahi_Social_LSTM
#from models.Model_Linear import Model_Linear
from models.Model_LSTM_1L import Model_LSTM_1L
from models.Model_LSTM_Scene import *

from utils.evaluate import *
from utils.data_loader import *
from utils.visualize import *
from utils.scene_grids import get_nonlinear_grids, get_common_grids, get_scene_states, update_scene_states
from config import *
from sample import sample


#Define model 
model= Model_LSTM_Scene

net = model(args, train = True)
if(args.use_cuda): net = net.cuda() 
optimizer = optim.RMSprop(net.parameters(), lr = args.learning_rate)
print(net)

# Define logger 
logger = Logger(args, train = True) # make logging utility
logger.write("{}\n".format(args))
# Load Data object
print("loading data...")
data_loader = DataLoader(args, logger, train = True)

input("Are you sure to train this network ? Enter")

#--------------------------------------------------------------------------------------------$
#                             TRAINING SECTION
#--------------------------------------------------------------------------------------------$

# Initialize scene states for all dataset.
# Intilize the grid-cell memories of 2 types: non-linear and common movments.
scene_states = {
    "nonlinear_h0_list": torch.zeros(args.num_datasets,args.num_grid_cells**2,args.num_layers, 1, args.rnn_size),
    "nonlinear_c0_list": torch.zeros(args.num_datasets,args.num_grid_cells**2,args.num_layers, 1, args.rnn_size),
    "common_h0_list" : torch.zeros(args.num_datasets,args.num_grid_cells**2,args.num_layers, 1, args.rnn_size),
    "common_c0_list": torch.zeros(args.num_datasets,args.num_grid_cells**2,args.num_layers, 1, args.rnn_size)
}
if(args.use_cuda): 
    scene_states["nonlinear_h0_list"], = scene_states["nonlinear_h0_list"].cuda() 
    scene_states["nonlinear_c0_list"]  = scene_states["nonlinear_c0_list"].cuda()
    scene_states["common_h0_list"]     = scene_states["common_h0_list"].cuda()
    scene_states["common_c0_list"]     = scene_states["common_c0_list"].cuda()

# Define scene information
scene_info ={ 
    "nonlinear_grid_list": [] ,
    "nonlinear_sub_grids_maps" : [], 
    "common_grid_list" : [], 
    "common_sub_grids_maps" : []
}

if(args.nonlinear_grids):
    logger.write("finding nonlinear grid_cells...")
    scene_info["nonlinear_grid_list"], scene_info["nonlinear_sub_grids_maps"] = get_nonlinear_grids(data_loader, args)
    logger.write("{}\n".format(scene_info["nonlinear_grid_list"]))

if(args.num_common_grids > 0):
    logger.write("finding common grid_cells...")
    scene_info["common_grid_list"], scene_info["common_sub_grids_maps"] = get_common_grids(data_loader, args)
    logger.write("{}\n".format(scene_info["common_grid_list"]))


# Load best trained stage1 model for training stage 2
if(args.stage2):
    save_model_file = os.path.join(args.save_prev_model_dir, "best_epoch_model.pt")
    print("loading best trained model at {}".format(save_model_file))
    state = torch.load(save_model_file)
    net.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])

    # Initialize scene states for all dataset.
    scene_states = state['scene_states']
    scene_info = state['scene_info']

    

# Init
best_MSE_validation = 100000; 
best_epoch = 10000000
start_epoch = 0

for e in range(start_epoch, args.nepochs):

    # Intialize variables
    loss_epoch = 0 
    MSE_validation = 0 
    MSE_train = 0 

    for i in range(0,data_loader.num_train_batches):
        start = time.time()

        # Load batch training data 
        batch  = data_loader.next_batch(randomUpdate=True)
        nPeds = batch["ped_ids"].size
        dataset_id = batch["dataset_id"]
        net.current_dataset = dataset_id

        # Init hidden states for each target in this batch
        h0 = Variable(torch.zeros(args.num_layers, nPeds, args.rnn_size))
        c0 = Variable(torch.zeros(args.num_layers, nPeds, args.rnn_size))
        if(args.use_cuda): 
            h0,c0 = h0.cuda(), c0.cuda()      
          
        # Clear things before forwarding a batch
        optimizer.zero_grad()
        batch_loss = 0 

        for t in range(args.observe_length + args.predict_length):

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
            scene_grid_h0, scene_grid_c0 = get_scene_states(xabs, dataset_id, scene_states,  scene_info, args)
            net.set_scene_hidden_states(scene_grid_h0, scene_grid_c0)                # Set hidden states for this model

            # set others model parameters  
            net.batch_size = batch["ped_ids_frame"][t].size
          
            # Loss of  this frame 
            loss_t = net.calculate_loss(xoff, xoff_next, xabs, xabs_next, batch["ped_ids_frame"][t], batch["ped_ids_frame"][t+1])
            batch_loss = batch_loss + loss_t

            # Store target's hidden states back to storage
            h0[:,indices,:] = net.i_h0.clone()
            c0[:,indices,:] = net.i_c0.clone()

            # Update scene states
            scene_grid_h0, scene_grid_c0 = net.get_scene_hidden_states()
            scene_states = update_scene_states(xabs, dataset_id, scene_states, scene_info, scene_grid_h0, scene_grid_c0, args)


        # Loss of batch and updating parameters 
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm(net.parameters(), args.grad_clip)
        optimizer.step()

        # Record loss for each epoch 
        loss_epoch = loss_epoch + batch_loss.item()
        stop = time.time()

        if (data_loader.num_train_batches* e + i) % args.info_freq == 0:

            logger.write('iter, {} / {} , loss_batch, {:3f},  time/iter , {:3f} ms'\
                .format(data_loader.num_train_batches* e + i, 
                    data_loader.num_train_batches*args.nepochs ,
                    loss_epoch/(i+1),
                    (stop-start)*1000))

    loss_epoch = loss_epoch/data_loader.num_train_batches

    # SAVE EPOCH MODEL
    save_model_file = '{}/net_epoch_{:06d}.pt'.format(args.save_model_dir,e)
    state = {
        'epoch': e,
        'scene_states': scene_states,
        'scene_info': scene_info,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state,save_model_file)   
    logger.write("saved model to " +  save_model_file)

#--------------------------------------------------------------------------------------------$
#                             VALIDATION SECTION 
#--------------------------------------------------------------------------------------------$

    logger.write('Calculating MSE on validation data ......')
    mse, _ , _ = sample(model, data_loader, save_model_file, args, validation = True)

    # Copy to best model if having better mse 
    if(mse < best_MSE_validation): 
         best_MSE_validation = mse
         best_epoch = e

         # Save best_model file
         best_epoch_model_dir = '{}/best_epoch_model.pt'.format(args.save_model_dir)
         shutil.copy(save_model_file,best_epoch_model_dir)

    # Stop time  
    end = time.time()

    # Print out results
    logger.write('epoch, {}, loss_epoch, {} , MSE(validation), {}'\
        .format( e, loss_epoch , mse), record_loss = True)
    
    logger.write('best_epoch, {}, MSE_best(test), {}'\
        .format( best_epoch, best_MSE_validation ))
    
