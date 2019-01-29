# LSTM
python train.py --model_dataset 0 --train_dataset 1 2 3 4 --predict_distance
python train.py --model_dataset 1 --train_dataset 0 2 3 4 --predict_distance 
python train.py --model_dataset 2 --train_dataset 0 1 3 4 --predict_distance
python train.py --model_dataset 3 --train_dataset 0 1 2 4 --predict_distance
python train.py --model_dataset 4 --train_dataset 0 1 2 3 --predict_distance



# nonlinear + common c16 (with  no subgrids map)
python train.py --model_dataset 0 --train_dataset 1 2 3 4 --predict_distance --num_common_grids 16 
python train.py --model_dataset 1 --train_dataset 0 2 3 4 --predict_distance --num_common_grids 16 
python train.py --model_dataset 2 --train_dataset 0 1 3 4 --predict_distance --num_common_grids 16 
python train.py --model_dataset 3 --train_dataset 0 1 2 4 --predict_distance --num_common_grids 16 
python train.py --model_dataset 4 --train_dataset 0 1 2 3 --predict_distance --num_common_grids 16 


