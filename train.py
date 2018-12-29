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
from config import *
from grid import get_allow_grids
from sample import *

''' usage: 
    python train.py --model_dataset 0 --train_dataset 1 2 3 4 --predict_distance   // for Model_LSTM_1L
    python train.py --model_dataset 0 --train_dataset 1 2 3 4 --predict_distance --use_scene --nonlinear_grids // for Model_LSTM_Scene non-linear
    python train.py --model_dataset 0 --train_dataset 1 2 3 4 --predict_distance --use_scene --all_grids  // for Model_LSTM_Scene all grids

    stage 2: 
    //all grids 
    python train.py --model_dataset 0 --train_dataset 0 --predict_distance --use_scene --all_grids --stage2 --nepochs 10
    // non-linear grids 
    python train.py --model_dataset 0 --train_dataset 0 --predict_distance --use_scene --nonlinear_grids --stage2 --nepochs 10
    // non-grids  
    python train.py --model_dataset 0 --train_dataset 0 --predict_distance --stage2 --nepochs 10

'''

# Select model 
model= Model_LSTM_Scene
model_dir = "Model_LSTM_Scene_non_linear_stage1_meters"
#previous_model_dir = "Model_LSTM_Scene_I_Sce_Hard"

# Parsing paramters from config file 
args.log_dir = './save/{}/v{}/log'.format(model_dir, args.model_dataset)
args.save_model_dir =  './save/{}/v{}/model'.format(model_dir,args.model_dataset)
args.num_train_datasets = len(args.train_dataset)
args.num_total_datasets = 5

# Parameters
best_MSE_validation = 100000; 
best_epoch = 10000000

# Define logger 
logger = Logger(args, train = True) # make logging utility
logger.write("{}\n".format(args))

# Load Data object
print("loading data...")
data_loader = DataLoader(args, logger, train = True)

#--------------------------------------------------------------------------------------------$
#                             TRAINING SECTION
#--------------------------------------------------------------------------------------------$
#Define model 
net = model(args, train = True)
if(args.use_cuda): net = net.cuda() 
optimizer = optim.RMSprop(net.parameters(), lr = args.learning_rate)
print(net)

# Initialize scene states for all dataset.
if(args.use_scene):
    scene_h0_list = Variable(torch.zeros(args.num_total_datasets,args.scene_grid_num*args.scene_grid_num,args.num_layers, 1, args.rnn_size))
    scene_c0_list = Variable(torch.zeros(args.num_total_datasets,args.scene_grid_num*args.scene_grid_num,args.num_layers, 1, args.rnn_size))
    if(args.use_cuda):
        scene_h0_list, scene_c0_list = scene_h0_list.cuda(), scene_c0_list.cuda()
            
    # Define which grids in each dataset are trained
    # allow_grid_list ~ list (size 5) of grid ids used to train
    allow_grid_list = get_allow_grids(data_loader,args)
    net.allow_grid_list = allow_grid_list

# Load a previous trained model 
if(args.stage2):
    previous_save_model_dir = './save/{}/v{}/model'.format(previous_model_dir,args.model_dataset)
    save_model_file = '{}/best_epoch_model.pt'.format(previous_save_model_dir)
    print("loading best trained model at {}".format(save_model_file))
    state = torch.load(save_model_file)
    net.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    start_epoch = 0
    
    # Initialize scene states for all dataset.
    if(args.use_scene and not args.stage2):
        scene_h0_list = state['scene_h0_list'].clone()
        scene_c0_list = state['scene_c0_list'].clone()    

else: 
    start_epoch = 0

input("Are you sure to train this network ? Enter")
for e in range(start_epoch, args.nepochs):

    # Intialize variables
    loss_epoch = 0 
    MSE_validation = 0 
    MSE_train = 0 

    for i in range(0,data_loader.num_train_batches):
        start = time.time()

        # Load batch training data 
        batch  = data_loader.next_batch(randomUpdate=True)

        # Get hidden states for each target in this batch
        nPeds = batch["ped_ids"].size
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
            if(args.use_scene):
                scene_grid_h0, scene_grid_c0, list_of_grid_id = get_scene_states(xabs, batch["dataset_id"], \
                                         args.scene_grid_num, scene_h0_list, scene_c0_list, args)
                net.set_scene_hidden_states(scene_grid_h0, scene_grid_c0)                # Set hidden states for this model
                net.current_grid_list = list_of_grid_id 
                net.current_dataset = batch["dataset_id"] 


            # set others model parameters  
            net.batch_size = batch["ped_ids_frame"][t].size
          
            # Loss of  this frame 
            loss_t = net.calculate_loss(xoff, xoff_next, xabs, xabs_next, batch["ped_ids_frame"][t], batch["ped_ids_frame"][t+1])
            batch_loss = batch_loss + loss_t

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
    logger.write("saved model to " +  save_model_file)

    if(args.use_scene):
        state = {
            'epoch': e,
            'scene_h0_list': scene_h0_list,
            'scene_c0_list': scene_c0_list,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict()
        }
    else:
        state = {
            'epoch': e,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict()
        }

    torch.save(state,save_model_file)   
    

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
    logger.write('epoch, {}, loss_epoch, {} , MSE(test), {}'\
        .format( e, loss_epoch , mse), record_loss = True)
    
    logger.write('best_epoch, {}, MSE_best(test), {}'\
        .format( best_epoch, best_MSE_validation ))
    
