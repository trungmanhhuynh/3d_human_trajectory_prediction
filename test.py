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
from models.Model_LSTM import Model_LSTM
from models.Model_LSTM_Scene_common_subgrids import Model_LSTM_Scene_common_subgrids 
from utils.evaluate import *
from utils.data_loader import *
from utils.visualize import *
from config import *
from sample import *

def get_model(args):

    model_dict={
        "Model_LSTM": Model_LSTM,
        "Model_LSTM_Scene_common" : Model_LSTM_Scene_common,
        "Model_LSTM_Scene_common_subgrids" : Model_LSTM_Scene_common_subgrids
        #"Model_LSTM_Scene_common_subgrids" : Model_LSTM_Scene_common_subgrids,
        #"Model_LSTM_Scene_common_subgrids_nonlinear" : Model_LSTM_Scene_common_subgrids_nonlinear
    }
    return model_dict[args.model_name]

if __name__ == '__main__':

    args = get_args()          # Get input argurments 
    args.max_datasets = 5      # Maximum number of sequences could be used for storing scene data
    args.log_dir = os.path.join(args.save_root , args.dataset_size, args.model_dir, str(args.model_dataset), 'log')
    args.save_model_dir =  os.path.join(args.save_root , args.dataset_size, args.model_dir, str(args.model_dataset), 'model')

    logger = Logger(args, train = False)                 # make logging utility
    logger.write("{}\n".format(args))
    data_loader = DataLoader(args, logger, train = False)
    model = get_model(args)

    logger.write('evaluating on test data ......')
    save_model_file = '{}/best_epoch_model.pt'.format(args.save_model_dir)
    mse_eval, nde_eval, fde_eval = sample(model, data_loader, save_model_file, args, test = True)

    # Print out results
    logger.write('mse_eval: {:.3f}, nde_eval: {:.3f}, fde_eval: {:.3f}'.format(mse_eval, nde_eval, fde_eval))