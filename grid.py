'''
Helper functions to compute the masks relevant to social grid

Author : Anirudh Vemula
Date : 29th October 2016
'''
import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt


def getGridMask(x_t, neighborhood_size, grid_size):
    
    # x_t is location at time t 
    # x_t ~ [batch_size,2]

    # Maximum number of pedestrians
    num_peds = x_t.size(0)
    frame_mask =  Variable(torch.zeros((num_peds, num_peds, grid_size**2))).cuda()

    # For each ped in the frame (existent and non-existent)
    for pedindex in range(num_peds):

        # Get x and y of the current ped
        current_x, current_y = x_t[pedindex, 0].data[0], x_t[pedindex, 1].data[0]

        width_low, width_high = current_x - neighborhood_size/2, current_x + neighborhood_size/2
        height_low, height_high = current_y - neighborhood_size/2, current_y + neighborhood_size/2

        # For all the other peds
        for otherpedindex in range(num_peds):

            # If the other pedID is the same as current pedID
            # The ped cannot be counted in his own grid
            if (otherpedindex == pedindex) :
                continue

            # Get x and y of the other ped
            other_x, other_y = x_t[otherpedindex, 0].data[0], x_t[otherpedindex, 1].data[0]
            if other_x >= width_high or other_x < width_low or other_y >= height_high or other_y < height_low:
                # Ped not in surrounding, so binary mask should be zero
                continue

            # If in surrounding, calculate the grid cell
            cell_x = int(np.floor(((other_x - width_low)/neighborhood_size) * grid_size))
            cell_y = int(np.floor(((other_y - height_low)/neighborhood_size) * grid_size))

            if cell_x >= grid_size or cell_x < 0 or cell_y >= grid_size or cell_y < 0:
                continue

            # Other ped is in the corresponding grid cell of current ped
            frame_mask[pedindex, otherpedindex, cell_x + cell_y*grid_size] = 1

    return frame_mask


def get_scene_states(x_t, dataset_id, scene_grid_num, scene_h0_list, scene_c0_list, args):
    
    # x_t is location at time t 
    # x_t ~ [batch_size,2]
    # scene_h0_list is list of grid hidden states
    # scene_c0_list is list of grid cell states 

    # Maximum number of pedestrians
    num_peds = x_t.size(0)
    rnn_size = scene_h0_list.size(4) # ~ [num_datasets,grid_Size**2,num_layers,batch_size,rnn_Size]
    num_layers = scene_h0_list.size(2)
    scene_grid_c0 =  Variable(torch.zeros((num_layers,num_peds, rnn_size)))  
    scene_grid_h0 =  Variable(torch.zeros((num_layers,num_peds, rnn_size)))  
    list_of_grid_id = Variable(torch.LongTensor(num_peds))
    if(args.use_cuda):
        list_of_grid_id, scene_grid_h0, scene_grid_c0 = list_of_grid_id.cuda(), scene_grid_h0.cuda(), scene_grid_c0.cuda() 

    # For each ped in the frame (existent and non-existent)
    for pedindex in range(num_peds):

        # Get x and y of the current ped
        current_x, current_y = x_t[pedindex, 0].data[0], x_t[pedindex, 1].data[0]

        width_low, width_high = -1 , 1          #scene is in range [-1,1]
        height_low, height_high = -1 , 1        #scene is in range [-1,1]
        boundary_size = 2 
  
        # calculate the grid cell
        cell_x = int(np.floor(((current_x - width_low)/boundary_size) * scene_grid_num))
        cell_y = int(np.floor(((current_y - height_low)/boundary_size) * scene_grid_num))

        # Peds locations must be in range of [-1,1], so the cell used must be in range [0,scene_grid_num-1]
        if(cell_x < 0):
            cell_x = 0
        if(cell_x >= scene_grid_num):
            cell_x = scene_grid_num - 1
        if(cell_y < 0):
            cell_y = 0 
        if(cell_y >= scene_grid_num):
            cell_y = scene_grid_num - 1

        list_of_grid_id[pedindex] = cell_x + cell_y*scene_grid_num

    scene_grid_h0 = torch.index_select(scene_h0_list[dataset_id],0,list_of_grid_id)
    scene_grid_c0 = torch.index_select(scene_c0_list[dataset_id],0,list_of_grid_id)

    scene_grid_c0 = scene_grid_c0.view(-1,num_peds,rnn_size)
    scene_grid_h0 = scene_grid_c0.view(-1,num_peds,rnn_size)

    return scene_grid_h0, scene_grid_c0, list_of_grid_id



def getGridMaskInference(frame, dimensions, neighborhood_size, grid_size):
    mnp = frame.shape[0]
    width, height = dimensions[0], dimensions[1]

    frame_mask = np.zeros((mnp, mnp, grid_size**2))

    width_bound, height_bound = (neighborhood_size/(width*1.0))*2, (neighborhood_size/(height*1.0))*2

    # For each ped in the frame (existent and non-existent)
    for pedindex in range(mnp):
        # Get x and y of the current ped
        current_x, current_y = frame[pedindex, 0], frame[pedindex, 1]

        width_low, width_high = current_x - width_bound/2, current_x + width_bound/2
        height_low, height_high = current_y - height_bound/2, current_y + height_bound/2

        # For all the other peds
        for otherpedindex in range(mnp):
            # If the other pedID is the same as current pedID
            if otherpedindex == pedindex:
                # The ped cannot be counted in his own grid
                continue

            # Get x and y of the other ped
            other_x, other_y = frame[otherpedindex, 0], frame[otherpedindex, 1]
            if other_x >= width_high or other_x < width_low or other_y >= height_high or other_y < height_low:
                # Ped not in surrounding, so binary mask should be zero
                continue

            # If in surrounding, calculate the grid cell
            cell_x = int(np.floor(((other_x - width_low)/width_bound) * grid_size))
            cell_y = int(np.floor(((other_y - height_low)/height_bound) * grid_size))

            if cell_x >= grid_size or cell_x < 0 or cell_y >= grid_size or cell_y < 0:
                continue
            
            # Other ped is in the corresponding grid cell of current ped
            frame_mask[pedindex, otherpedindex, cell_x + cell_y*grid_size] = 1

    return frame_mask

def getSequenceGridMask(sequence, dimensions, neighborhood_size, grid_size, using_cuda):
    '''
    Get the grid masks for all the frames in the sequence
    params:
    sequence : A numpy matrix of shape SL x MNP x 3
    dimensions : This will be a list [width, height]
    neighborhood_size : Scalar value representing the size of neighborhood considered
    grid_size : Scalar value representing the size of the grid discretization
    using_cuda: Boolean value denoting if using GPU or not
    '''
    sl = len(sequence)
    sequence_mask = []

    for i in range(sl):
        # sequence_mask[i, :, :, :] = getGridMask(sequence[i, :, :], dimensions, neighborhood_size, grid_size)
        mask = Variable(torch.from_numpy(getGridMask(sequence[i], dimensions, neighborhood_size, grid_size)).float())
        if using_cuda:
            mask = mask.cuda()
        sequence_mask.append(mask)

    return sequence_mask

def is_nonlinear(traj):
        non_linear_degree = 0

        y_max = traj[-1,1]
        y_min = traj[0,1]
        y_s   =traj[int(traj.shape[0]/2),1]
        if((y_max-y_min)/2 + y_min == 0):
            non_linear_degree = 0
        else:
            non_linear_degree = abs(((y_max - y_min)/2 + y_min - y_s))

        if(non_linear_degree >= 0.2): 
            return True
        else: 
            return False

def get_nonlinear_trajectories(batch,args):

    nonlinear_traj = []
    # Get trajectories for each frame
    for ped in batch["ped_ids"]: 
        traj = [] 
        for t in range(0,args.observe_length + args.predict_length):
            pid_idx = np.where(batch["ped_ids_frame"][t] == ped)[0]
            traj.append(batch["batch_data_absolute"][t][pid_idx])
        
        traj = np.vstack(traj[:])
        if(traj.shape[0] <= 5):
            continue

        # Check if this traj is non-linear 
        if(is_nonlinear(traj)): 
            nonlinear_traj.append(traj)

    return nonlinear_traj

def get_allow_grids(data_loader, args):

    allow_grid_list = [[] for i in range(5)]

    # DEFAULT: use all grids
    if(args.non_grids):
        # still empty 
        return allow_grid_list

    if(args.all_grids):
        for d in range(args.num_total_datasets):
            allow_grid_list[d] = np.arange(0,args.scene_grid_num**2).tolist()

    if(args.manual_grids):
        allow_grid_list[0] = [26,27,28,29,30,34,35,36,37,38]
        allow_grid_list[1] = [10,12,13,51,52,59,60]
        allow_grid_list[2] = [0,1,13,14,15,41,42,58,59]
        allow_grid_list[3] = [30,31,36,37,38,39,24]
        allow_grid_list[4] = [30,31,36,37,38,39,24]

    if(args.nonlinear_grids):

        allow_grid_list[0] = [36, 37, 38, 39, 42, 43, 44, 50, 29, 45, 51, 52, 21, 22, 14]
        allow_grid_list[1] = [9, 10, 18, 27, 35, 43, 51, 28, 36, 19, 59, 20, 44, 52]
        allow_grid_list[2] = [43, 44, 52, 53, 61, 62, 45, 36, 9, 10, 18, 19, 27, 28, 29, 30,\
                             31, 39, 47, 55, 34, 35, 42, 33, 50, 58, 59, 60, 51, 41, 37, 38, \
                             40, 54, 63, 46, 32, 21, 22, 14, 6, 26, 25, 15, 20, 23, 17, 24, 13]
        allow_grid_list[3] = [32, 33, 41, 48, 56, 42, 44, 45, 50, 51, 52, 46, 23, 31, 36, 37,\
                              38, 39, 22, 30, 28, 27]
        allow_grid_list[4] = [34, 35, 36, 44, 52, 53, 61, 14, 15, 22, 23, 28, 29, 30, 37, 38, \
                              27, 26, 24, 32, 40, 31, 39, 45, 46, 25, 33, 41, 60, 47, 48, 49, \
                              56, 5, 13, 20, 21, 16]

        ''' UNCOMMENT THIS FOR REAL CALCULATION, THAT PRODUCES ABOUT RESULTS
        # Find all non-linear trajectories and which dataset these non-linear
        # trajectories belongs to 
        for i in range(0,data_loader.num_train_batches):
            # Load batch training data 
            batch  = data_loader.next_batch(randomUpdate=False)
            # Extract non-linear trajectories 
            nonlinearTraj = get_nonlinear_trajectories(batch,args)
            
            #Save nonlinear trajectories
            fig = plt.clf()
            for j in range(len(nonlinearTraj)):
                plt.plot(nonlinearTraj[0][:,0], nonlinearTraj[0][:,1],'ro-')
                plt.axis([-1, 1, -1, 1])
                plt.savefig("./non_linear_trajectories/v{}_b{}_tr{}.png".format(batch["dataset_id"],i,j))
                plt.close()

            # Find which grids, the non linear trajectories pass by 
            if(len(nonlinearTraj) > 0 ):
                list_of_grid_id = find_nonlinear_grids(nonlinearTraj, batch ,args)
                for grid in list_of_grid_id:
                    if (grid not in allow_grid_list[batch["dataset_id"]]):
                        allow_grid_list[batch["dataset_id"]].append(grid)
            
            '''
    return allow_grid_list

def find_nonlinear_grids(nonlinearTraj, batch ,args):

    width_low, width_high = -1 , 1          #scene is in range [-1,1]
    height_low, height_high = -1 , 1        #scene is in range [-1,1]
    boundary_size = 2 
  
    list_of_grid_id = []

    # Process each trajectories    
    for tr in range(len(nonlinearTraj)):

        thisTraj = nonlinearTraj[tr]
        #  Find which grid each location belongs
        for loc in range(thisTraj.shape[0]):

            current_x, current_y = thisTraj[loc,0], thisTraj[loc,1]
            # calculate the grid cell
            cell_x = int(np.floor(((current_x - width_low)/boundary_size) * args.scene_grid_num))
            cell_y = int(np.floor(((current_y - height_low)/boundary_size) * args.scene_grid_num))

            # Peds locations must be in range of [-1,1], so the cell used must be in range [0,scene_grid_num-1]
            if(cell_x < 0):
                cell_x = 0
            if(cell_x >= args.scene_grid_num):
                cell_x = args.scene_grid_num - 1
            if(cell_y < 0):
                cell_y = 0 
            if(cell_y >= args.scene_grid_num):
                cell_y = args.scene_grid_num - 1

            list_of_grid_id.append(cell_x + cell_y*args.scene_grid_num)

    return np.unique(list_of_grid_id)




