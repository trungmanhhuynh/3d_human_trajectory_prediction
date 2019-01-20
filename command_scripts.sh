
### Stage 1
# lstm 
python train.py --model_dataset 0 --train_dataset 1 2 3 4 --predict_distance 
python train.py --model_dataset 1 --train_dataset 0 2 3 4 --predict_distance 
python train.py --model_dataset 2 --train_dataset 0 1 3 4 --predict_distance 
python train.py --model_dataset 3 --train_dataset 0 1 2 4 --predict_distance 
python train.py --model_dataset 4 --train_dataset 0 1 2 3 --predict_distance 

# nonlinear (no subgrids map)
python train.py --model_dataset 0 --train_dataset 1 2 3 4 --predict_distance --nonlinear_grids
python train.py --model_dataset 1 --train_dataset 0 2 3 4 --predict_distance --nonlinear_grids
python train.py --model_dataset 2 --train_dataset 0 1 3 4 --predict_distance --nonlinear_grids
python train.py --model_dataset 3 --train_dataset 0 1 2 4 --predict_distance --nonlinear_grids
python train.py --model_dataset 4 --train_dataset 0 1 2 3 --predict_distance --nonlinear_grids

# nonlinear (with subgrids map)
python train.py --model_dataset 0 --train_dataset 1 2 3 4 --predict_distance --nonlinear_grids --use_sub_grids_map
python train.py --model_dataset 1 --train_dataset 0 2 3 4 --predict_distance --nonlinear_grids --use_sub_grids_map
python train.py --model_dataset 2 --train_dataset 0 1 3 4 --predict_distance --nonlinear_grids --use_sub_grids_map
python train.py --model_dataset 3 --train_dataset 0 1 2 4 --predict_distance --nonlinear_grids --use_sub_grids_map
python train.py --model_dataset 4 --train_dataset 0 1 2 3 --predict_distance --nonlinear_grids --use_sub_grids_map

# common c16 (no subgrids map)
python train.py --model_dataset 0 --train_dataset 1 2 3 4 --predict_distance --num_common_grids 16
python train.py --model_dataset 1 --train_dataset 0 2 3 4 --predict_distance --num_common_grids 16
python train.py --model_dataset 2 --train_dataset 0 1 3 4 --predict_distance --num_common_grids 16
python train.py --model_dataset 3 --train_dataset 0 1 2 4 --predict_distance --num_common_grids 16
python train.py --model_dataset 4 --train_dataset 0 1 2 3 --predict_distance --num_common_grids 16

# common c16 (with  subgrids map)
python train.py --model_dataset 0 --train_dataset 1 2 3 4 --predict_distance --num_common_grids 16 --use_sub_grids_map
python train.py --model_dataset 1 --train_dataset 0 2 3 4 --predict_distance --num_common_grids 16 --use_sub_grids_map
python train.py --model_dataset 2 --train_dataset 0 1 3 4 --predict_distance --num_common_grids 16 --use_sub_grids_map
python train.py --model_dataset 3 --train_dataset 0 1 2 4 --predict_distance --num_common_grids 16 --use_sub_grids_map
python train.py --model_dataset 4 --train_dataset 0 1 2 3 --predict_distance --num_common_grids 16 --use_sub_grids_map

# nonlinear + common c16 (with  no subgrids map)
python train.py --model_dataset 0 --train_dataset 1 2 3 4 --predict_distance --nonlinear_grids --num_common_grids 16 
python train.py --model_dataset 1 --train_dataset 0 2 3 4 --predict_distance --nonlinear_grids --num_common_grids 16 
python train.py --model_dataset 2 --train_dataset 0 1 3 4 --predict_distance --nonlinear_grids --num_common_grids 16 
python train.py --model_dataset 3 --train_dataset 0 1 2 4 --predict_distance --nonlinear_grids --num_common_grids 16 
python train.py --model_dataset 4 --train_dataset 0 1 2 3 --predict_distance --nonlinear_grids --num_common_grids 16 


############################# Stage 2: 
# lstm  
python train.py --model_dataset 0 --train_dataset 0 --stage2 --predict_distance --nepochs 10
python train.py --model_dataset 1 --train_dataset 1 --stage2 --predict_distance --nepochs 10
python train.py --model_dataset 2 --train_dataset 2 --stage2 --predict_distance --nepochs 10
python train.py --model_dataset 3 --train_dataset 3 --stage2 --predict_distance --nepochs 10
python train.py --model_dataset 4 --train_dataset 4 --stage2 --predict_distance --nepochs 10

# common c16 (no subgrids map)
python train.py --model_dataset 0 --train_dataset 0 --stage2 --predict_distance --num_common_grids 16 --nepochs 10
python train.py --model_dataset 1 --train_dataset 1 --stage2 --predict_distance --num_common_grids 16 --nepochs 10
python train.py --model_dataset 2 --train_dataset 2 --stage2 --predict_distance --num_common_grids 16 --nepochs 10
python train.py --model_dataset 3 --train_dataset 3 --stage2 --predict_distance --num_common_grids 16 --nepochs 10
python train.py --model_dataset 4 --train_dataset 4 --stage2 --predict_distance --num_common_grids 16 --nepochs 10 

# common c16 (with subgrids map)
python train.py --model_dataset 0 --train_dataset 0 --stage2 --predict_distance --num_common_grids 16 --use_sub_grids_map --nepochs 10
python train.py --model_dataset 1 --train_dataset 1 --stage2 --predict_distance --num_common_grids 16 --use_sub_grids_map --nepochs 10
python train.py --model_dataset 2 --train_dataset 2 --stage2 --predict_distance --num_common_grids 16 --use_sub_grids_map --nepochs 10
python train.py --model_dataset 3 --train_dataset 3 --stage2 --predict_distance --num_common_grids 16 --use_sub_grids_map --nepochs 10
python train.py --model_dataset 4 --train_dataset 4 --stage2 --predict_distance --num_common_grids 16 --use_sub_grids_map --nepochs 10 

# nonlinear (no subgrids map)
python train.py --model_dataset 0 --train_dataset 0 --stage2 --predict_distance --nonlinear_grids --nepochs 10
python train.py --model_dataset 1 --train_dataset 1 --stage2 --predict_distance --nonlinear_grids --nepochs 10
python train.py --model_dataset 2 --train_dataset 2 --stage2 --predict_distance --nonlinear_grids --nepochs 10
python train.py --model_dataset 3 --train_dataset 3 --stage2 --predict_distance --nonlinear_grids --nepochs 10
python train.py --model_dataset 4 --train_dataset 4 --stage2 --predict_distance --nonlinear_grids --nepochs 10

# nonlinear (with subgrids map)
python train.py --model_dataset 0 --train_dataset 0 --stage2 --predict_distance --nonlinear_grids --use_sub_grids_map --nepochs 10
python train.py --model_dataset 1 --train_dataset 1 --stage2 --predict_distance --nonlinear_grids --use_sub_grids_map --nepochs 10
python train.py --model_dataset 2 --train_dataset 2 --stage2 --predict_distance --nonlinear_grids --use_sub_grids_map --nepochs 10
python train.py --model_dataset 3 --train_dataset 3 --stage2 --predict_distance --nonlinear_grids --use_sub_grids_map --nepochs 10
python train.py --model_dataset 4 --train_dataset 4 --stage2 --predict_distance --nonlinear_grids --use_sub_grids_map --nepochs 10

# nonlinear + common grids = 16 NO subgrids maps
python train.py --model_dataset 0 --train_dataset 0 --stage2 --predict_distance --nonlinear_grids --num_common_grids 16 --nepochs 10
python train.py --model_dataset 1 --train_dataset 1 --stage2 --predict_distance --nonlinear_grids --num_common_grids 16 --nepochs 10
python train.py --model_dataset 2 --train_dataset 2 --stage2 --predict_distance --nonlinear_grids --num_common_grids 16 --nepochs 10
python train.py --model_dataset 3 --train_dataset 3 --stage2 --predict_distance --nonlinear_grids --num_common_grids 16 --nepochs 10
python train.py --model_dataset 4 --train_dataset 4 --stage2 --predict_distance --nonlinear_grids --num_common_grids 16 --nepochs 10


# nonlinear + common grids = 16 with subgrids maps
python train.py --model_dataset 0 --train_dataset 0 --stage2 --predict_distance --nonlinear_grids --num_common_grids 16 --use_sub_grids_map --nepochs 10
python train.py --model_dataset 1 --train_dataset 1 --stage2 --predict_distance --nonlinear_grids --num_common_grids 16 --use_sub_grids_map --nepochs 10
python train.py --model_dataset 2 --train_dataset 2 --stage2 --predict_distance --nonlinear_grids --num_common_grids 16 --use_sub_grids_map --nepochs 10
python train.py --model_dataset 3 --train_dataset 3 --stage2 --predict_distance --nonlinear_grids --num_common_grids 16 --use_sub_grids_map --nepochs 10
python train.py --model_dataset 4 --train_dataset 4 --stage2 --predict_distance --nonlinear_grids --num_common_grids 16 --use_sub_grids_map --nepochs 10
