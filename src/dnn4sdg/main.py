import argparse
import time 
import os
import sys
import numpy as np
import importlib
from .model import init_model, train_model, compile_model
from .evaluation import evaluate_model
from .io_util import save_makedirs, save_model_summary, load_model, create_directories
from .geotiff_util import visualise_labels
#
from  .get_args import params_template,read_args,print_args

#--------------------------

def worker(args,config):

    print('---- args from worker ---')
    print_args(args)
    print('-------------------------')
    
    #create model dir if it does not exist
    create_directories([args.model_dir])

    hparams = dict(config.hyperparameters)
    
    if args.init_model:
        print('initializing model ..')
        model = init_model(args.patch_size, args.model_dir, **hparams)
        save_model_summary(config.hyperparameters, model, args.model_dir)
        
    elif args.train_model or args.evaluate_model:
        print('loading model ..')        
        model = load_model(args.model_dir)
        model = compile_model(model, hparams["learning_rate"],
                                  hparams['momentum'],
                                  hparams["decay"])

    if args.train_model:
        print('training model .. epochs=',args.epochs)        
        model = train_model( model,
            config.train_images,config.train_labels,
            args.patch_size,args.model_dir,
            nb_epoch = args.epochs,
            checkpoints = args.checkpoints,
            tensorboard = args.tensorboard,
            earlystop = args.earlystop)
    
    if args.evaluate_model:
        print('evaluating model ..')        
        evaluate_model(model,
            config.train_images,config.train_labels,                           
            args.patch_size,args.model_dir, out_format=args.out_format)

    return model

if __name__ == '__main__':

    argv=sys.argv[1:]

    #check if config file is passed    
    config_file=None
    for x in argv:
        if '--config' in x or '--ini' in x:
            config_file=x.split('=')[1]            
            print('config file is: ',config_file)
    
    #load config file if given
    if not config_file is None:
        cpath, cfile = os.path.split(config_file)
        sys.path.append(cpath)
        
        #import config file
        config = importlib.import_module(cfile.split('.')[0])

        #do the main computation here
        if config.args.debug:
            print('----- arguments: ')
            print(config.args)
            print('-----')
        else:
            model=worker(config.args,config)            
    else:
        print('----- No config file found! ---')        
        print('passed arguments are: ',argv)
        print()
        print('   Usage: python main.py --config <file> ...')        
   
