'''
Author : Huynh Manh
Date : 
'''

import os
import sys
import pickle
import numpy as np
import random
import math
import time

class DataLoader():

    def __init__(self, args, logger , train= False):

        # List of data directories where raw data resides
        self.dataset_dirs = ['./datasets/trajectory_prediction/data/eth_hotel', 
                             './datasets/trajectory_prediction/data/eth_univ' ,
                             './datasets/trajectory_prediction/data/ucy_univ' , 
                             './datasets/trajectory_prediction/data/ucy_zara01',
                             './datasets/trajectory_prediction/data/ucy_zara02']
             

        self.logger = logger           
        self.used_datasets = [0, 1, 2, 3, 4]
        self.input_metric = args.input_metric

        if(train):
            # all the train_dataset will be used to train the model
            # which then will be used for testing the model_dataset
            self.model_dataset = [args.model_dataset]
            self.train_dataset = args.train_dataset
            self.valid_dataset = args.train_dataset
            self.val_fraction =  0.2 # 20% batches used for validation  
            self.train_fraction =  0.8 #20% batches used for training  

            if(args.stage2):
                self.val_fraction =  0.5  # 50% batches used for validation  
                self.train_fraction =  0.5 #50% batches used for training  

        else:
            self.test_dataset = [args.test_dataset]
            self.test_fraction = 1 # 20% batches used for testing  

        self.used_data_dirs = [self.dataset_dirs[x] for x in self.used_datasets]
        self.num_datasets = len(self.used_data_dirs)                          # Number of datasets
        self.data_dir = './datasets/trajectory_prediction'                                       # Where the pre-processed pickle file resides             
        
        if(self.input_metric is "meters"):
            data_file = os.path.join(self.data_dir, "trajectories_meters.cpkl")          # Where the pre-processed pickle file resides
        elif(self.input_metric is "pixels"):
            data_file = os.path.join(self.data_dir, "trajectories_pixels.cpkl")
        else:
            print("UTIL:Invalid input metric") 
            sys.exit(0) 

        print("Dataset used :", self.used_data_dirs )
        # Assign testing flag 
        self.train = train
        self.pre_process = args.pre_process
        self.tsteps = args.observe_length + args.predict_length

        # 
        self.num_batches = 0 ; 
        self.num_train_batches = 0 ; 
        self.num_validation_batches = 0 ; 
        self.num_test_batches = 0 ; 


        # If the file doesn't exist
        if (not(os.path.exists(data_file)) or self.pre_process):
            print("Pre-processing data from raw data")
            self.frame_preprocess(self.used_data_dirs, data_file)         #Preprocess data file       

        # Load the processed data from the pickle file
        self.load_preprocessed(data_file)

        # Reset all the data pointers of the dataloader object
        self.reset_batch_pointer(train = True, valid = True , test = True)


    def frame_preprocess(self, data_dirs, data_file):
        '''
        This function splits video data into batches of frames. Each batch
        is a dictionary.
        
        + Each dataset will have its own set of batches, stored in all_data 
        
        '''

        # Intialize output values 
        all_batches = []     
        num_batches_list = []        # number baches of each dataset

        # Proceed each dataset at a time
        for dataset_id, directory in enumerate(data_dirs):
        
            all_batches.append([])

            # Load the data from the  file
            if(self.input_metric is "meters"):
                file_path = os.path.join(directory, 'data_normalized_meters_2.5fps.txt')
            elif(self.input_metric is "pixels"):
                file_path = os.path.join(directory, 'data_normalized_pixels_2.5fps.txt')
            else: 
                print("UTIL:Invalid input metric") 
                sys.exit(0) 

            data = np.genfromtxt(file_path, delimiter=',')

            # Get all frame numbers of the current dataset
            frameList = np.unique(data[:,0]).tolist()            

            # Read frame-by-frame and store them to a temporary data for this dataset 
            dataset_data = [] 
            for ind, frame in enumerate(frameList):

                # Extract all pedestrians in current frame
                pedDataInFrame = data[data[:,0] == frame,:]   # [frameId, targetId, x ,y]

                # Store it to temporary dataset_data
                dataset_data.append(pedDataInFrame)
            
            # number of frames of refined datasetdata
            nFrames = len(dataset_data)
            nBatches = nFrames - (self.tsteps + 1)
            self.logger.write("{}: {} --> {} frames ".format(directory , len(frameList), nFrames))

            for i in range(0,  nFrames - (self.tsteps + 1)):

                # Initialize batch as a dictionary 
                batch = {
                    "batch_data_absolute": [] ,          # location x,y of peds in each frame
                    "batch_data_offset": [] ,            # location x,y of peds in each frame
                    "ped_ids_frame": [],                 # ped ids in each frames
                    "ped_ids": [] ,                      # all ped ids in this batch
                    "frame_list": -1 ,                   # list of frame number of this batch
                    "dataset_id": -1                     # dataset id of this batch
                }

                # let me get all mixed data for the batch, then I will split them out
                # for better control of it. 
                temp_batch = dataset_data[i:i+self.tsteps+1]                #  size of tsteps + 1,
                                                                            # each has [frameId, targetId, x ,y]
                # Convert temp_batch to array for easier processing
                temp_batch = np.vstack(temp_batch)
                #np.set_printoptions(suppress=True)
                #print("temp_batch =", temp_batch)

                # Set start frame of this batch 
                batch["frame_list"] = np.unique(temp_batch[:,0])

                # Set dataset id of this batch 
                batch["dataset_id"] = dataset_id

                # Set all ped ids of this batch
                batch["ped_ids"] = np.unique(temp_batch[:,1])

                #np.set_printoptions(suppress=True)

                for ind, frameId in enumerate(batch["frame_list"]):
                    # Find all data index of this frame
                    frame_data = temp_batch[temp_batch[:,0] == frameId,:]

                    # Store x,y location 
                    batch["batch_data_absolute"].append(frame_data[:,[2,3]])        
            
                    # Store ped ids 
                    batch["ped_ids_frame"].append(frame_data[:,1])       

                # Get data_offset 
                batch_data_offset = self.calculate_batch_data_offset(batch)
                batch["batch_data_offset"] = batch_data_offset

                # End of processing a batch    
                # Gather into all batches
                all_batches[dataset_id].append(batch)

            # End of processing one dataset     
            num_batches_list.append(nBatches)
            self.logger.write("{}: number of batches {}".format(directory, len(all_batches[dataset_id])))

        self.num_batches = sum(num_batches_list)
        self.logger.write("Total number of batches: {}".format(self.num_batches))

        # Save the tuple (all_frame_data, frameList_data, numPeds_data) in the pickle file
        f = open(data_file, "wb")
        pickle.dump((all_batches, num_batches_list), f, protocol=2)
        f.close()

    def calculate_batch_data_offset(self, batch):
        

        def Cloning(li1):
            li_copy =[]
            for item in li1: li_copy.append(np.copy(item))
            return li_copy
 
        # Initialize batch_data_offset as batch_data_absolute
        batch_data_offset = Cloning(batch["batch_data_absolute"])

        # Process each target
        for ped in batch["ped_ids"]:
            for t in reversed(range(self.tsteps + 1)):
                if ped in batch["ped_ids_frame"][t]:
                    idx_loc_t =  batch["ped_ids_frame"][t].tolist().index(ped)

                    if(t == 0): # t is starting frame
                        batch_data_offset[t][idx_loc_t,:]  =  0
                    elif(ped in batch["ped_ids_frame"][t-1]): # t is not a starting frame
                        idx_loc_prev_t =  batch["ped_ids_frame"][t-1].tolist().index(ped)
                        batch_data_offset[t][idx_loc_t] =  batch_data_offset[t][idx_loc_t] -  batch_data_offset[t-1][idx_loc_prev_t]
                    else:   # t is starting frame
                        batch_data_offset[t][idx_loc_t,:]  =  0

        return batch_data_offset

    def load_preprocessed(self, data_file):
        '''
        Function to load the pre-processed data into the DataLoader object
        params:
        data_file : the path to the pickled data file
        '''
        # Load data from the pickled file
        f = open(data_file, 'rb')
        self.raw_data = pickle.load(f)
        f.close()

        # Get all the data from the pickle file
        self.all_batches = self.raw_data[0]
        self.num_batches_list = self.raw_data[1]

        # Intilizing list for each mode
        self.train_batch = [] 
        self.validation_batch = []
        self.test_batch = []
        self.num_train_batches_list = [0]*self.num_datasets
        self.num_validation_batches_list = [0]*self.num_datasets  
        self.num_test_batches_list = [0]*self.num_datasets  

        # For each dataset
        for dataset in range(self.num_datasets):
            self.train_batch.append([]) 
            self.validation_batch.append([])
            self.test_batch.append([])
            
            if(self.train == True):

                if(dataset in self.train_dataset):

                    # get the train data for the specified dataset
                    #self.num_train_batches_list[dataset] = int(self.num_batches_list[dataset]*(1 - self.val_fraction))
                    self.num_train_batches_list[dataset] = int(self.num_batches_list[dataset]*self.train_fraction)
                    self.train_batch[dataset] = self.all_batches[dataset][0:self.num_train_batches_list[dataset]]
                    
                    # get the validation data for the specified dataset
                    self.num_validation_batches_list[dataset] = int(self.num_batches_list[dataset]*self.val_fraction)
                    self.validation_batch[dataset]  = self.all_batches[dataset][-self.num_validation_batches_list[dataset]:]
        
            else:
                if(dataset in self.test_dataset):

                    self.num_test_batches_list[dataset] = int(self.num_batches_list[dataset]*self.test_fraction)
                    self.test_batch[dataset]  = self.all_batches[dataset][-self.num_test_batches_list[dataset]:]

            self.logger.write('---')
            self.logger.write('Training data from dataset {} :{}'.format(dataset, len(self.train_batch[dataset])))
            self.logger.write('Validation data from dataset {} :{}'.format(dataset, len(self.validation_batch[dataset]) ))
            self.logger.write('Test data from dataset {} :{}'.format(dataset, len(self.test_batch[dataset]) ))

        self.logger.write('---')
        self.num_train_batches = sum(self.num_train_batches_list)
        self.num_validation_batches= sum(self.num_validation_batches_list)
        self.num_test_batches = sum(self.num_test_batches_list)
        self.logger.write('Total num_train_batches : {}'.format(sum(self.num_train_batches_list)))
        self.logger.write('Total num_validation_batches : {}'.format(sum(self.num_validation_batches_list)))
        self.logger.write('Total num_test_batches : {}'.format(sum(self.num_test_batches_list)))

    def next_batch(self, randomUpdate=True):
        '''
        Function to get the next batch of points
        '''
        # Extract the frame data of the current dataset
        dataset_idx =self.train_dataset[self.train_dataset_pointer]
    
        # Get the frame pointer for the current dataset
        batch_idx = self.train_batch_pointer
    
        # Number of unique peds in this sequence of frames
        batch_data = self.train_batch[dataset_idx][batch_idx]

        # advance the frame pointer to a random point
        if randomUpdate:
            self.train_dataset_pointer = random.randint(0, len(self.train_dataset) -1 )
            dataset_idx = self.train_dataset[self.train_dataset_pointer]
            self.train_batch_pointer = random.randint(0, self.num_train_batches_list[dataset_idx] -1)         
        else:
            
            self.tick_batch_pointer(train=True)

        return batch_data

    def next_valid_batch(self, randomUpdate=False):
        '''
        Function to get the next batch of points
        '''

       # Extract the frame data of the current dataset
        dataset_idx =self.train_dataset[self.valid_dataset_pointer]

        # Get the frame pointer for the current dataset
        batch_idx = self.valid_batch_pointer
    
        # Number of unique peds in this sequence of frames
        batch_data = self.validation_batch[dataset_idx][batch_idx]

        self.tick_batch_pointer(valid=True)
   
        return batch_data

    def next_test_batch(self, randomUpdate=False):
        '''
        Function to get the next batch of points
        '''
        # Extract the frame data of the current dataset
        dataset_idx =self.test_dataset[self.test_dataset_pointer]
        # Get the frame pointer for the current dataset
        batch_idx = self.test_batch_pointer
    
        # Number of unique peds in this sequence of frames
        batch_data = self.test_batch[dataset_idx][batch_idx]

        self.tick_batch_pointer(test=True)
   
        return batch_data

    def tick_batch_pointer(self, train = False, valid = False , test = False):
        '''
        Advance the dataset pointer
        '''
        if train:
            self.train_batch_pointer += 1                   # Increment batch pointer
            dataset_idx =self.train_dataset[self.train_dataset_pointer]
            if self.train_batch_pointer  >= self.num_train_batches_list[dataset_idx] :
                # Go to the next dataset
                self.train_dataset_pointer += 1
                # Set the frame pointer to zero for the current dataset
                self.train_batch_pointer = 0
                # If all datasets are done, then go to the first one again
                if self.train_dataset_pointer >= len(self.train_dataset):
                    self.reset_batch_pointer(train = True)
        if valid:       
            self.valid_batch_pointer += 1                   # Increment batch pointer
            dataset_idx =self.train_dataset[self.valid_dataset_pointer]
            if self.valid_batch_pointer  >= self.num_validation_batches_list[dataset_idx] :
                # Go to the next dataset
                self.valid_dataset_pointer += 1
                # Set the frame pointer to zero for the current dataset
                self.valid_batch_pointer = 0
                # If all datasets are done, then go to the first one again
                if self.valid_dataset_pointer >= len(self.valid_dataset):
                    self.reset_batch_pointer(valid = True)

        if test:       
            self.test_batch_pointer += 1              # Increment batch pointer
            dataset_idx =self.test_dataset[self.test_dataset_pointer]
            if self.test_batch_pointer  >= self.num_test_batches_list[dataset_idx] :
                # Go to the next dataset
                self.test_dataset_pointer += 1
                # Set the frame pointer to zero for the current dataset
                self.test_batch_pointer = 0
                # If all datasets are done, then go to the first one again
                if self.test_dataset_pointer >= len(self.test_dataset):
                    self.reset_batch_pointer(test = True)



    def reset_batch_pointer(self, train = False, valid = False , test = False):
        '''
        Reset all pointers
        '''
        if train:
            # Go to the first frame of the first dataset
            self.train_dataset_pointer = 0
            self.train_batch_pointer = 0

        if valid:
            self.valid_dataset_pointer = 0
            self.valid_batch_pointer = 0

        if test:
            self.test_dataset_pointer = 0
            self.test_batch_pointer = 0

# abstraction for logging
class Logger():
    def __init__(self, args , train = False):
        
        self.train = train
        if(train):
            # open file for record screen 
            self.train_screen_log_file_path = '{}/train_screen_log.txt'.format(args.log_dir) 
            self.train_screen_log_file = open(self.train_screen_log_file_path, 'w')
           
            # open file for recording train loss 
            self.train_log_file_path = '{}/train_log.txt'.format(args.log_dir) 
            self.train_log_file= open(self.train_log_file_path, 'w')

        else: 
            self.test_screen_log_file_path = '{}/test_screen_log.txt'.format(args.log_dir) 
            self.test_screen_log_file = open(self.test_screen_log_file_path, 'w')

    def write(self, s, record_loss = False):

        print (s)

        if(self.train):
            with open(self.train_screen_log_file_path, 'a') as f:
                f.write(s + "\n")
            if(record_loss):
                with open(self.train_log_file_path, 'a') as f:
                    f.write(s + "\n")
        else:
            with open(self.test_screen_log_file_path, 'a') as f:
                f.write(s + "\n")


