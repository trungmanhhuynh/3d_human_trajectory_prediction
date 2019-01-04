import argparse


parser = argparse.ArgumentParser()

# General 
parser.add_argument('--pre_process',action="store_true", default=False, help='pre-preprocess data')
parser.add_argument('--use_cuda',action="store_true", default=False, help='using CUDA ?')
parser.add_argument('--load_best_train',action="store_true", default=False, help='load the best trained model')
parser.add_argument('--input_metric', type=str, default= 'pixels', help='specify input metric(meters or pixels)')
parser.add_argument('--output_metric', type=str, default= 'meters', help='specify output metric(meters or pixels)')

# training params
parser.add_argument('--rnn_size', type=int, default= 128, help='size of RNN hidden state')
parser.add_argument('--num_layers', type=int, default=1, help=' LSTM layers ')
parser.add_argument('--embedding_size', type=int, default=64, help='size of embedding layer ')
parser.add_argument('--nmixtures', type=int, default= 1, help='number of gaussian mixtures')
parser.add_argument('--input_size', type=int, default=2, help='size of one input data')
parser.add_argument('--nepochs', type=int, default= 50, help='number of epochs')
parser.add_argument('--dropout', type=float, default=0.2, help='probability of keeping neuron during dropout')
parser.add_argument('--grad_clip', type=float, default=10., help='clip gradients to this magnitude')
parser.add_argument('--optimizer', type=str, default='rmsprop', help="ctype of optimizer: 'rmsprop' 'adam'")
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--decay', type=float, default=0.95, help='decay rate for rmsprop')
parser.add_argument('--momentum', type=float, default=1, help='momentum for rmsprop')

parser.add_argument('--predict_distance',action="store_true", default=False,  help='use offset data as input')

parser.add_argument('--train_dataset', nargs ="*" ,type=int, default=[0 ,1 ,2, 3, 4], help=' this dataset for training ')
parser.add_argument('--model_dataset', type=int, default= 0, help=' this dataset for training ')

# Social 
parser.add_argument('--add_social', action="store_true", default=False, help='add social model')
parser.add_argument('--neighborhood_size', type=float, default=0.2, help=' size of bounding box')
parser.add_argument('--social_grid_size', type=int, default=4, help='number of grids')
parser.add_argument('--social_mixtures', type=int, default= 1, help='number of mixtures output from Social LSTM')

#Scene 
parser.add_argument('--stage2',action="store_true", default=False, help='specify if it is stage 2 training')
parser.add_argument('--use_scene', action="store_true", default=False , help='add scene model')
parser.add_argument('--non_grids', action="store_true", default=False , help='no grids are trained')
parser.add_argument('--nonlinear_grids', action="store_true", default=False , help='nonlinear grids are trained')
parser.add_argument('--scene_lstm_8', action="store_true", default=False , help='nonlinear grids are trained')
parser.add_argument('--scene_lstm_16', action="store_true", default=False , help='nonlinear grids are trained')
parser.add_argument('--scene_lstm_32', action="store_true", default=False , help='nonlinear grids are trained')
parser.add_argument('--scene_lstm_64', action="store_true", default=False , help='nonlinear grids are trained')
parser.add_argument('--scene_lstm_n8', action="store_true", default=False , help='nonlinear grids are trained')
parser.add_argument('--scene_lstm_n16', action="store_true", default=False , help='nonlinear grids are trained')
parser.add_argument('--scene_lstm_n32', action="store_true", default=False , help='nonlinear grids are trained')
parser.add_argument('--scene_lstm_nU16', action="store_true", default=False , help='nonlinear grids are trained')


parser.add_argument('--scene_mixtures', type=int, default= 1, help='number of mixtures output from Scene LSTM')
parser.add_argument('--scene_grid_num', type=int, default= 8, help='number of grids of scene')
parser.add_argument('--inner_grid_num', type=int, default= 8, help='number of grids inside one grid')



# testing params
parser.add_argument('--test_dataset', type=int, default= 0, help=' this dataset for training ')
parser.add_argument('--tepochs', type=int, default= 0, help='number of epochs used to load checkpointfile')
parser.add_argument('--observe_length', type=int, default= 8, help='number of obseved frames')
parser.add_argument('--predict_length', type=int, default= 12, help='number of predicted frames')
parser.add_argument('--meters', action="store_true", default=False, help='error units')


#book-keeping
parser.add_argument('--data_dir', type=str, default='./data', help='location, relative to execution, of data')
parser.add_argument('--save_freq', type=int, default= 1, help='')
parser.add_argument('--info_freq', type=int, default=10, help='frequency to print out')

args = parser.parse_args()
