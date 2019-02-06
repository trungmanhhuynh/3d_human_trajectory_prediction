import random
import math
import torch
import argparse
import time
import tqdm
import shutil
import os 

from torch import nn, optim
from torch.autograd import Variable
from matplotlib import pyplot as plt
import numpy as np

from models.Model_Linear import Model_Linear
from models.Model_LSTM import Model_LSTM
from models.Model_LSTM_abs import Model_LSTM_abs
from models.Model_LSTM_Scene import Model_LSTM_Scene
from models.Model_LSTM_Scene_SoftFilter import Model_LSTM_Scene_SoftFilter
from models.Model_LSTM_Scene_nonlinear import Model_LSTM_Scene_nonlinear 
from models.Model_LSTM_Scene_nonlinear_noSF import Model_LSTM_Scene_nonlinear_noSF 
from models.Model_LSTM_Scene_nonlinear_subgrids import Model_LSTM_Scene_nonlinear_subgrids
from models.Model_LSTM_Scene_nonlinear_subgrids_noSF import Model_LSTM_Scene_nonlinear_subgrids_noSF
from models.Social_LSTM_abs import Social_LSTM_abs

from utils.evaluate import  *
from utils.data_loader import DataLoader, Logger
from utils.visualize import *
from config import get_args
from sample import sample


def run_model(model, data_loader, logger, args):

    #Create model 
    torch.manual_seed(1)
    net = model(data_loader, args, train = True)
    if(args.use_cuda): net = net.cuda() 
    optimizer = optim.RMSprop(net.parameters(), lr = args.learning_rate)
    #for param in net.parameters():
    #    print(param.data)
    print(net)
    #input("here")

    #--------------------------------------------------------------------------------------------$
    #                             TRAINING SECTION
    #--------------------------------------------------------------------------------------------$

    # Load best trained stage1 model for training stage 2
    if(args.stage2):
        save_model_file = os.path.join(args.save_model_dir, "best_epoch_model.pt")
        print("loading best trained model at {}".format(save_model_file))
        state = torch.load(save_model_file)
        net.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
     
    best_mse_val, best_epoch, start_epoch= 10e5, 10e5, 0  
    for e in range(start_epoch, args.nepochs):
        epoch_loss = 0                                                           # Init loss of this epoch
        data_loader.shuffle_data()                                               # Shuffle data
        for i in range(0,data_loader.num_train_batches):
            # forward/backward for each train batch
            start = time.time()                                                  # Start timer
            batch  = data_loader.next_batch(randomUpdate=True)                  # Load batch training data 
            net.init_batch_parameters(batch)                                     # 
            net.init_target_hidden_states()                                      # Init hidden states for each target in this batch
            optimizer.zero_grad()                                                # Zero out gradients
            batch_loss = 0                                                       # Init loss of this batch

            for t in range(args.observe_length + args.predict_length - 1):
            
                # Get batch data at time t and t+1
                cur_frame = {"loc_off": batch["loc_off"][t], "loc_abs": batch["loc_abs"][t], "frame_pids":  batch["all_pids"]}
                next_frame = {"loc_off": batch["loc_off"][t+1],"loc_abs": batch["loc_abs"][t+1], "frame_pids":  batch["all_pids"]}

                # Process current batch and produce loss
                loss_t = net.process(cur_frame, next_frame)
                batch_loss = batch_loss + loss_t

            # back ward
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            optimizer.step()
            epoch_loss = epoch_loss + batch_loss.item()/(args.observe_length + args.predict_length) # Add batch_loss to epoch loss
            stop = time.time()                                                  # Stop timer

            # logging
            if (data_loader.num_train_batches* e + i) % args.info_freq == 0:
                logger.write('iter:{}/{}, batch_loss:{:f}, time/iter:{:.3f} ms, time left: {:.3f} hours' \
                            .format(data_loader.num_train_batches* e + i, data_loader.num_train_batches*args.nepochs,
                            epoch_loss/(i+1),  (stop-start)*1000,
                            (data_loader.num_train_batches*args.nepochs - data_loader.num_train_batches* e + i)*(stop-start)/3600))


        epoch_loss = epoch_loss/data_loader.num_train_batches                   # Loss of this epoch

        # Save model in each epoch
        if(args.stage2):
            save_model_file = '{}/net_stage2_epoch_{:06d}.pt'.format(args.save_model_dir,e)
        else: 
            save_model_file = '{}/net_epoch_{:06d}.pt'.format(args.save_model_dir,e)

        state = {'epoch': e, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()}
        if(args.use_scene):
            state['scene_states'] = net.scene_states 
            state['scene_info']   = net.scene_info    

        torch.save(state,save_model_file)   
        logger.write("saved model to " +  save_model_file)

    #--------------------------------------------------------------------------------------------$
    #                             VALIDATION SECTION 
    #--------------------------------------------------------------------------------------------$

        logger.write('Calculating MSE on validation data ......')
        mse_eval, nde_eval, fde_eval = sample(model, data_loader, save_model_file, args, validation = True)

        # Copy to best model if having better mse_eval 
        if(mse_eval < best_mse_val): 
            best_mse_val, best_epoch = mse_eval, e 
            best_nde_eval = nde_eval
            best_fde_eval = fde_eval
            if(args.stage2):             
                best_epoch_model_dir = '{}/best_epoch_model_stage2.pt'.format(args.save_model_dir)          
            else :
                best_epoch_model_dir = '{}/best_epoch_model.pt'.format(args.save_model_dir)          
            shutil.copy(save_model_file,best_epoch_model_dir)                  # Save best_model file

        # Print out results
        logger.write('epoch: {}, epoch_loss: {}, mse_eval: {:.3f}'.format(e, epoch_loss , mse_eval), record_loss = True)
        logger.write('best_epoch: {}, mse_eval: {:.3f}, nde_eval: {:.3f}, fde_eval: {:.3f}'\
                    .format(best_epoch, best_mse_val, best_nde_eval, best_fde_eval))
        

def get_model(args):
    model_dict={
        "Model_Linear":Model_Linear,
        "Model_LSTM": Model_LSTM,
        "Model_LSTM_abs": Model_LSTM_abs,
        "Model_LSTM_Scene": Model_LSTM_Scene,
        "Model_LSTM_Scene_SoftFilter": Model_LSTM_Scene_SoftFilter,
        "Model_LSTM_Scene_nonlinear" : Model_LSTM_Scene_nonlinear,
        "Model_LSTM_Scene_nonlinear_noSF": Model_LSTM_Scene_nonlinear_noSF,
        "Model_LSTM_Scene_nonlinear_subgrids" : Model_LSTM_Scene_nonlinear_subgrids,
        "Model_LSTM_Scene_nonlinear_subgrids_noSF" : Model_LSTM_Scene_nonlinear_subgrids_noSF,
        "Social_LSTM_abs": Social_LSTM_abs
    }
    if(model_dict[args.model_name] is Model_LSTM or model_dict[args.model_name] is Model_LSTM_abs or 
        model_dict[args.model_name] is Social_LSTM_abs or model_dict[args.model_name] is Model_Linear):
        assert args.use_nonlinear_grids is False
        assert args.use_subgrid_maps is False 
        assert args.use_scene is False
    if(model_dict[args.model_name] is Model_LSTM_Scene or model_dict[args.model_name] is Model_LSTM_Scene_SoftFilter):
        assert args.use_nonlinear_grids is False
        assert args.use_subgrid_maps is False 
        assert args.use_scene is True
    if(model_dict[args.model_name] is Model_LSTM_Scene_nonlinear or model_dict[args.model_name] is Model_LSTM_Scene_nonlinear_noSF):
        assert args.use_nonlinear_grids is True    
        assert args.use_subgrid_maps is False 
        assert args.use_scene is True
    if(model_dict[args.model_name] is Model_LSTM_Scene_nonlinear_subgrids or model_dict[args.model_name] is Model_LSTM_Scene_nonlinear_subgrids_noSF):
        assert args.use_nonlinear_grids is True    
        assert args.use_subgrid_maps is True   
        assert args.use_scene is True

    return model_dict[args.model_name]

if __name__ == '__main__':

    args = get_args()          # Get input argurments 
    args.max_datasets = 5      # Maximum number of sequences could be used for storing scene data
    args.log_dir = os.path.join(args.save_root , args.dataset_size, args.model_dir, str(args.model_dataset), 'log')
    args.save_model_dir =  os.path.join(args.save_root , args.dataset_size, args.model_dir, str(args.model_dataset), 'model')
    
    logger = Logger(args, train = True)                 # make logging utility
    logger.write("{}\n".format(args))

    # Stage 1: train on 4 sequences, test on the remaining. 
    # e.g. --model_dataset 0 --train_dataset 1 2 3 4 
    logger.write("------training stage1-----------")
    args.stage2 = False 
    data_loader = DataLoader(args, logger, train = True)
    model = get_model(args)
    run_model(model, data_loader, logger, args)

    # Stage 2: train on 50% of model dataset and test on the remaining 50% 
    #. e.g. --model_dataset 0 --train_dataset 0 
    logger.write("------training stage2-----------")
    args.train_dataset = [args.model_dataset]                   # train on part of model sequence.
    args.stage2 = True 
    args.nepochs = 10 
    # logger = Logger(args, train = True)                       # re-create Logger
    data_loader = DataLoader(args, logger, train = True)        # re-create data_loader
    run_model(model, data_loader, logger, args)
