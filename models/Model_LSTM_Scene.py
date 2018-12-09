'''
 Scene Model with Gates
 Author: Huynh Manh

'''
import random
import math
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from matplotlib import pyplot as plt
from bs4 import BeautifulSoup as Soup
import numpy as np
import glob
from grid import *

def logsumexp(x):
    x_max, _ = x.max(dim=1,keepdim=True)
    x_max_expand = x_max.expand(x.size())
    res =  x_max + torch.log((x-x_max_expand).exp().sum(dim=1, keepdim=True))
    return res

class Model_LSTM_Scene(nn.Module):
    def __init__(self, args, train = False):
        super(Model_LSTM_Scene, self).__init__()

        #-----General Parameters
        self.use_cuda = args.use_cuda
        self.nmixtures = args.nmixtures                                    # number of mixtures in final output
        self.rnn_size= args.rnn_size       # rnn_size of ALL LSTMs
        self.predict_length = args.predict_length  
        self.observe_length = args.observe_length
        self.tsteps = args.observe_length + args.predict_length            # number of steps 
        self.num_layers = args.num_layers                                  # number of layers for ALL LSTMs
        self.embedding_size = args.embedding_size                          # size of embedding layers for ALL 
        self.output_size = self.nmixtures*6                                # final output size 
        self.train = train 
        self.predict_distance = args.predict_distance                       # set train/test flag

        if(train): 
            self.dropout = args.dropout 
        else: 
            self.dropout = 0       # set dropout value
           
        #-----Scene Parameters    
        self.scene_grid_num = args.scene_grid_num                         # number of grid size for scene
        self.inner_grid_num = args.inner_grid_num
        self.scene_mixtures = args.scene_mixtures 
        self.scene_output_size = self.scene_mixtures*6                      # number of grid size for scene
        self.allow_grid_list = []
        self.current_grid_list = 0 
        self.current_dataset = 0
        # Scene mode
        self.non_grids = args.non_grids 
        self.all_grids = args.all_grids
        self.manual_grids = args.manual_grids
        self.nonlinear_grids = args.nonlinear_grids

        self.Sigmoid = nn.Sigmoid() 
        self.ReLU = nn.ReLU() 
        self.Tanh = nn.Tanh() 

        #---- Scene Models
        # convert absolute location to one-hot locations 
        self.LSTM_Scene = nn.LSTM(self.inner_grid_num**2 + self.rnn_size, self.rnn_size, num_layers=1, dropout=self.dropout)

        #--- Indivudal model
        self.Embedding_Input = nn.Linear(2,self.embedding_size)
        self.I_LSTM_L1    = nn.LSTM(self.embedding_size, self.rnn_size, num_layers=1, dropout=self.dropout)

        # Modify individual LSTM
        self.F_gate = nn.Linear(self.rnn_size + self.embedding_size, self.rnn_size)

        self.Final_Output    = nn.Linear(self.rnn_size, self.output_size)

    def set_target_hidden_states(self, h0_t, c0_t): 
        
        # Init hidden/cell states for Layer 1: 
        self.i_h0 = h0_t.clone()
        self.i_c0 = c0_t.clone()

    def set_scene_hidden_states(self, scene_grid_h0, scene_grid_c0):

        # In case we use trained hidden states
        self.scene_grid_h0 = scene_grid_h0.clone()
        self.scene_grid_c0 = scene_grid_c0.clone()

    # Forward bacth data at time t
    def forward(self, xoff, xabs):
 
        # Get one-hot vectors of locations
        # scene_embedding ~ [1, batch_size,inner_grid_size**2]

        scene_embedding = self.get_onehot_location(xabs, self.batch_size)

        input_scene =  torch.cat([scene_embedding,self.i_h0], dim = 2)

        lstm_scene_output, (self.scene_grid_h0, self.scene_grid_c0) = self.LSTM_Scene(input_scene, (self.scene_grid_h0, self.scene_grid_c0))
        lstm_scene_output = self.ReLU(lstm_scene_output)

        # hard-filter scene grids
        for idx, grid_id in enumerate(self.current_grid_list.data):
            if(grid_id not in self.allow_grid_list[self.current_dataset]):
                lstm_scene_output[:,idx,:] = Variable(torch.zeros(self.num_layers, 1, self.rnn_size)).clone()

        #----- Individual Movememts
        if(self.predict_distance):
            embedding_input = self.Embedding_Input(xoff.unsqueeze(0))
        else: 
            embedding_input = self.Embedding_Input(xabs.unsqueeze(0))

        #embedding_input = self.ReLU(embedding_input)

        i_lstm_output_l1,(self.i_h0, self.i_c0) = self.I_LSTM_L1(embedding_input,(self.i_h0, self.i_c0))
        #i_lstm_output_l1 = self.ReLU(i_lstm_output_l1)

        # which information of the scene should be use to 
        # adjust individual movements ?  
        filter_input = torch.cat([embedding_input,i_lstm_output_l1], dim = 2)
        filter_gate =  self.F_gate(filter_input)
        filter_gate =  self.Sigmoid(filter_gate) 
        filtered_scene_lstm = filter_gate*lstm_scene_output             # element-wise multiplication

        # Modify the individual movements by filtered scene
        i_lstm_output_l1 = i_lstm_output_l1 + filtered_scene_lstm
        i_lstm_output_l1 = self.ReLU(i_lstm_output_l1)

        # Convert to final output 
        final_output = self.Final_Output(i_lstm_output_l1)

        final_output = final_output.view(-1,self.output_size)
        # split output into pieces. Each ~ [1*batch_size, nmixtures]
        # and store them to final results
        mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits = final_output.split(self.nmixtures,dim=1)   


        return mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits

    def calculate_loss(self, xoff, xoff_next, xabs, xabs_next, ped_ids_t, ped_ids_tplus1):

        # Each ~ [batch_size, nmixtures]
        mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits = self.forward(xoff, xabs)

        # Which data should be used to calculate loss ? 
        # Only caculate for peds that presents in both frames 
        indices = np.where(ped_ids_t[:, None] == ped_ids_tplus1[None, :])
        
        # if there is no same peds present in next frame
        # then return loss = 0 ?
        if(indices[0].size ==0): return 0 

        indices_t = Variable(torch.LongTensor(indices[0]))
        indices_tplus1 = Variable(torch.LongTensor(indices[1]))
        if(self.use_cuda):
            indices_t,indices_tplus1 = indices_t.cuda(), indices_tplus1.cuda()

        # Use indices to select which peds's location used for calculating loss
        mu1  = torch.index_select(mu1,0,indices_t)
        mu2  = torch.index_select(mu2,0,indices_t)
        log_sigma1  = torch.index_select(log_sigma1,0,indices_t)
        log_sigma2  = torch.index_select(log_sigma2,0,indices_t)
        rho  = torch.index_select(pi_logits,0,indices_t)
        pi_logits  = torch.index_select(pi_logits,0,indices_t)

        # x1, x2 ~ [batch_size, 1]   
        xabs_next, xoff_next  = xabs_next.view(-1,2) , xoff_next.view(-1,2) 
        if(self.predict_distance):
            x1, x2 = xoff_next.split(1,dim=1)
        else:
            x1, x2 = xabs_next.split(1,dim=1)

        x1  = torch.index_select(x1,0,indices_tplus1)
        x2  = torch.index_select(x2,0,indices_tplus1)

        loss = - self.logP_gaussian(x1, x2, mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits)

        return loss/self.batch_size 


    def logP_gaussian(self,x1, x2, mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits):
        x1, x2 = x1.repeat(1,self.nmixtures), x2.repeat(1,self.nmixtures)
        sigma1, sigma2 = log_sigma1.exp(), log_sigma2.exp()
        rho = nn.functional.tanh(rho)
        log_pi = nn.functional.log_softmax(pi_logits, dim = 1)
        z_tmp1, z_tmp2 = (x1-mu1)/sigma1, (x2-mu2)/sigma2
        z = z_tmp1**2 + z_tmp2**2 - 2*rho*z_tmp1*z_tmp2
        # part one
        log_gaussian = - math.log(math.pi*2)-log_sigma1 - log_sigma2 - 0.5*(1-rho**2).log()
        # part two
        log_gaussian += - z/2/(1-rho**2)
        # part three
        log_gaussian = logsumexp(log_gaussian + log_pi)
        return log_gaussian.sum()


    def get_best_location(self, mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits):
            
        batch_size = mu1.size(0)

        # x_best ~ [batch_size, 2]
        x_best = Variable(torch.zeros(batch_size,2))
        if(self.use_cuda):
            x_best = x_best.cuda()
            
        for i in range(batch_size):

            pi = nn.functional.softmax(pi_logits[i],dim =0)

            idx, = random.choices(range(self.nmixtures), weights = pi.data.tolist())
        
            x_best[i,0] = mu1[i,idx]
            x_best[i,1] = mu2[i,idx] 
    
        return x_best 

    def get_onehot_location(self, x_t, batch_size):

        # x_t ~ [1,batch_size, 2]

        # Initialize one-hot location
        one_hot = Variable(torch.zeros(1, batch_size, self.inner_grid_num**2))
        if(self.use_cuda):
            one_hot = one_hot.cuda()

        # For each ped in the frame
        for pedindex in range(batch_size):

            # Get x and y of the current ped
            current_x, current_y = x_t[pedindex, 0].data[0], x_t[pedindex, 1].data[0]

            width_low, width_high = -1 , 1          #scene is in range [-1,1]
            height_low, height_high = -1 , 1        #scene is in range [-1,1]
            boundary_size = 2 
      
            # calculate the grid cell
            cell_x = int(np.floor(((current_x - width_low)/boundary_size) * self.inner_grid_num))
            cell_y = int(np.floor(((current_y - height_low)/boundary_size) * self.inner_grid_num))

            # Peds locations must be in range of [-1,1], so the cell used must be in range [0,scene_grid_num-1]
            if(cell_x < 0):
                cell_x = 0
            if(cell_x >= self.inner_grid_num):
                cell_x = self.inner_grid_num - 1
            if(cell_y < 0):
                cell_y = 0 
            if(cell_y >= self.inner_grid_num):
                cell_y = self.inner_grid_num - 1

            one_hot[:,pedindex,cell_y*self.inner_grid_num + cell_x] = 1 

        return one_hot

