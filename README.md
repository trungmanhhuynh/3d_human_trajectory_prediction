
# Human-Trajectory-Prediction
Using 2d-grid scene lstm with hard and soft filter to improve individual human trajectory prediction.


### Prerequisites

 + Pytorch 
 + Python 3.4 


# Before Training/Testing Your Models

 Run following command to create directories for each model
 ```
 >>  sh make_dir.sh               
 ```
 Because we train each model in 2 stages, so we also need to create directories for stage 2 of each model.
 In make_dir.sh, change {model_name}_stage_1 to {model_name}_stage_2 and run above command again.
 Applying the same concept, you can also create directories for other models/configurations.

### Test Using Pre-trained Model 


### Train Your Own Models: 
 
 ```
  python train.py --model_dataset 4 --train_dataset 0 1 2 3 --use_scene --use_distance --nonlinear_grids
 ```
 Please read the config.py for further details of each flag.



