import argparse
import time 
import os
import sys
import numpy as np

__all__=["params_template","print_args","arg_parser","read_args"]

avail_models=["one_layer","two_layer",
              "one_layer_pool_2","one_layer_pool_3",
              "one_layer_pool_4","one_layer_filter_12",
              "one_layer_filter_16","one_layer_filter_20",
              "two_layer_pool_2","two_layer_pool_3","two_layer_pool_4",
              "two_layer_filter_3", "two_layer_filter_4",
              "two_layer_filter_5", "two_layer_filter_6"]

class params_template():
        
    def __init__(self):
        self.preprocess_data=False
        self.init_model=True
        self.train_model=True
        self.evaluate_model=True
        self.debug=False
        self.architecture=avail_models[1]
        self.visualise=False
        self.tensorboard=True
        self.checkpoint=True
        self.oldformat=False
        self.earlystop=False
        self.dataset=False
        self.patch_size=64
        self.epochs=20
        self.model_dir=None
        self.ofraction=20
        self.setup=False
        self.out_format='GeoTIFF'
        
    def __str__(self):
        for x in dir(self):
            print('{}={}'.format(x,getattr(self, x)))
            
    def get_var(self,name):
        #call the methods defined above by name
        getattr(foo, name)()

default_args=params_template()

def print_args(args):
    for x in dir(args):        
        val=getattr(args, x)
        if not (x.startswith('__') or x.startswith('_')):
            if isinstance(val, (list,int,np.ndarray,float,str)):
                print('{}={}'.format(x,val))

def arg_parser(default=default_args):
    info='''
    Train a convolutional neural network 
    to predict building in  satellite images
    '''
    parser = argparse.ArgumentParser(description=info)

    #check first of config is given
    parser.add_argument('--ini','--config',
                        dest='config',
                        default=None,
                        help='configuration file.')   
    
    parser.add_argument(
        "-p", "--preprocess-data",
        dest="preprocess_data",
        action="store_const",
        const=True,
        default=default.preprocess_data,
        help="When selected preprocess data.")
    
    parser.add_argument(
        "-i", "--init-model",
        dest="init_model",
        action="store_const",
        const=True,
        default=default.init_model,
        help="When selected initialise model.")
    
    parser.add_argument(
        "-t", "--train-model",
        dest="train_model",
        action="store_const",
        const=True,
        default=default.train_model,
        help="When selected train model.")
    
    parser.add_argument(
        "-e", "--evaluate-model",
        dest="evaluate_model",
        action="store_const",
        const=True,
        default=default.evaluate_model,
        help="When selected evaluate model.")
    
    parser.add_argument(
        "-d", "--debug",
        dest="debug",
        action="store_const",
        const=True,
        default=default.debug,
        help="Run on a small test dataset.")
    
    parser.add_argument(
        "-a", "--architecture",
        dest="architecture",
        choices=avail_models,
        default=default.architecture,        
        help="Neural net architecture.")
    
    parser.add_argument(
        "-v", "--visualise",
        dest="visualise",
        action="store_const",
        const=True,
        default=default.visualise,
        help="Visualise labels.")
    
    parser.add_argument(
        "-T", "--tensorboard",
        dest="tensorboard",
        action="store_const",
        const=True,
        default=default.tensorboard,
        help="Store tensorboard data while training.")
    
    parser.add_argument(
        "-C", "--checkpoints",
        dest="checkpoints",
        action="store_const",
        const=True,
        default=default.checkpoint,
        help="Create checkpoints while training.")

    parser.add_argument(
        "--oldformat",
        dest="oldformat",
        action="store_const",
        const=True,
        default=default.oldformat,
        help="use old data reading format.")
    
    parser.add_argument(
        "-E", "--earlystop",
        dest="earlystop",
        action="store_const",
        const=True,
        default=default.earlystop,
        help="Create earlystop while training.")
    
    parser.add_argument(
        "--dataset",
        choices=["sentinel"],
        default=default.dataset,        
        help="Determine which dataset to use.")
    
    parser.add_argument(
        "--patch-size",
        dest="patch_size",        
        default=default.patch_size,
        type=int,
        help="Choose the patch size.")
    
    parser.add_argument(
        "--epochs",
        default=default.epochs,
        type=int,
        help="Number of training epochs.")
    
    parser.add_argument(
        "--model-id",
        dest="model_dir",        
        default=default.model_dir,
        type=str,
        help="Model that should be used. must be an existing ID.")
    
    parser.add_argument(
        "--ofrac",
        dest='ofraction',
        default = default.ofraction,
        type=float,
        help="Percentage of building distribution in the image")
        
    parser.add_argument(
        "--setup",        
        default=default.setup,
        action="store_const",
        const=True,
        help="Create all necessary directories for the classifier to work.")
    
    parser.add_argument(
        "--out-format",
        dest='out_format',
        default=default.out_format,
        choices=["GeoTIFF", "Shapefile"],
        help="Determine the format of the output for the evaluation method.")
    
    return parser


def read_args(default=default_args):
    parser = arg_parser(default)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = read_args()
    print(args)
    
