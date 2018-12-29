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
    python test.py --test_dataset 0 --predict_distance   // for Model_LSTM_1L
    python test.py --test_dataset 0 --predict_distance --use_scene --nonlinear_grids  
    python test.py --test_dataset 0 --predict_distance --use_scene --all_grids

'''

# Select model 
model= Model_LSTM_1L
model_dir = "Model_LSTM_1L_stage1_pixels"

# ---Parsing paramters from config file 
args.log_dir = './save/{}/v{}/log'.format(model_dir, args.test_dataset)
args.save_model_dir =  './save/{}/v{}/model'.format(model_dir,args.test_dataset)
args.save_test_result_pts_dir ='./save/{}/v{}/test_result_pts'.format(model_dir,args.test_dataset)
#args.save_train_result_pts_dir ='./save/v{}/LSTM_1L_scene_distance_v2/train_result_pts'.format(args.test_dataset)
#args.save_trajectory_gaussian_dir ='../save_scene_lstm/run0/video0/lstm_scene_social/res_gaussian'


# --- Define logger 
logger = Logger(args, train = False) # make logging utility
logger.write("{}\n".format(args))

# --- Load Data object
print("loading data...")
data_loader = DataLoader(args, logger, train = False)

#--- Load a trained model 
save_model_file = '{}/best_epoch_model.pt'.format(args.save_model_dir)
print("loading best trained model at {}".format(save_model_file))

#--- Calculate mse for test data
mse, nonLinearMSE, fde = sample(model, data_loader, save_model_file, args, test = True)

#--- Print out results
logger.write('Testing: mse = {}, non-linear mse = {}, fde = {}'.format(mse, nonLinearMSE, fde))


