# LSTM
python train.py --model_dataset 0 --train_dataset 1 2 3 4
python train.py --model_dataset 1 --train_dataset 0 2 3 4
python train.py --model_dataset 2 --train_dataset 0 1 3 4
python train.py --model_dataset 3 --train_dataset 0 1 2 4
python train.py --model_dataset 4 --train_dataset 0 1 2 3

# nonlinear
python train.py --model_dataset 0 --train_dataset 1 2 3 4 --use_nonlinear_grids
python train.py --model_dataset 1 --train_dataset 0 2 3 4 --use_nonlinear_grids
python train.py --model_dataset 2 --train_dataset 0 1 3 4 --use_nonlinear_grids
python train.py --model_dataset 3 --train_dataset 0 1 2 4 --use_nonlinear_grids
python train.py --model_dataset 4 --train_dataset 0 1 2 3 --use_nonlinear_grids


# nonlinear subgrids
python train.py --model_dataset 0 --train_dataset 1 2 3 4 --use_nonlinear_grids --use_subgrid_maps
python train.py --model_dataset 1 --train_dataset 0 2 3 4 --use_nonlinear_grids --use_subgrid_maps
python train.py --model_dataset 2 --train_dataset 0 1 3 4 --use_nonlinear_grids --use_subgrid_maps
python train.py --model_dataset 3 --train_dataset 0 1 2 4 --use_nonlinear_grids --use_subgrid_maps
python train.py --model_dataset 4 --train_dataset 0 1 2 3 --use_nonlinear_grids --use_subgrid_maps



python test.py --model_dataset 0 --num_common_grids 16 --use_subgrid_maps
