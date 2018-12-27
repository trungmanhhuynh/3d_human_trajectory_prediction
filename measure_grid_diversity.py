# Description: Function to measure grid diversity - the levels of chaotic motion
# in each grid-cell of each video sequences. The outputs is list of grid-cell in order
# of diversity levels. 
# Author: Manh Huynh
# Date: 12/24/2018 

import math
import torch
import argparse
import time
import tqdm
import shutil
import sys

from torch import nn, optim
from torch.autograd import Variable
from matplotlib import pyplot as plt
import numpy as np
from utils.data_loader import *
from config import *

def measure_grid_diversity(): 

    number_grid_cells = 64 ;         # 8x8 grid_cells
    number_sub_grids = 64 ;          # 8x8 number inner grid_ceslls in each grid 


    # Load data 
    model_dir = "Model_LSTM_1L_stage1_meters"
    args.log_dir = './save/{}/v{}/log'.format(model_dir, args.test_dataset)
    args.save_model_dir =  './save/{}/v{}/model'.format(model_dir,args.test_dataset)
    logger = Logger(args, train = False) # make logging utility
    data_loader = DataLoader(args, logger, train = False)

    # a grid-cell is further divided into sub-grids, 
    # each sub-grid has 8 directions to nearyby grids. 
    grid_cell = {
        "sub_grids": np.zeros(number_sub_grids),               
        "direction_subgrids": np.zeros((number_sub_grids,9))       # 9 possible moving directions in each sub-grid
    }
    list_grid_cells = [grid_cell for i in range(number_grid_cells)]

    for i in range(0, data_loader.num_test_batches):

        batch  = data_loader.next_test_batch(randomUpdate=False)

        print(batch)
        input("here")


# Test function
if __name__ == '__main__':
  #test_convert_normalized_pixels_to_meters()
  measure_grid_diversity()
