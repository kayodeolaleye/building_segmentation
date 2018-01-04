import argparse
import time 
import os
import sys
import fiona
from folders import DATASETS, OUTPUT_DIR, TRAIN_DATA_DIR, LABELS_DIR
from preprocessing import preprocess_data
from model import init_model, train_model, compile_model
from evaluation import evaluate_model
from io_util import save_makedirs, save_model_summary, load_model, create_directories
from geotiff_util import visualise_labels
import numpy as np

np.random.seed(123)

def create_parser():
    parser = argparse.ArgumentParser(description="Train a convolutional neural network to predict building in  satellite images")
    
    parser.add_argument(
        "-p", "--preprocess-data",
        dest="preprocess_data",
        action="store_const",
        const=True,
        default=False,
        help="When selected preprocess data.")
    
    parser.add_argument(
        "-i", "--init-model",
        dest="init_model",
        action="store_const",
        const=True,
        default=False,
        help="When selected initialise model.")
    
    parser.add_argument(
        "-t", "--train-model",
        dest="train_model",
        action="store_const",
        const=True,
        default=False,
        help="When selected train model.")
    
    parser.add_argument(
        "-e", "--evaluate-model",
        dest="evaluate_model",
        action="store_const",
        const=True,
        default=False,
        help="When selected evaluate model.")
    
    parser.add_argument(
        "-d", "--debug",
        dest="debug",
        action="store_const",
        const=True,
        default=False,
        help="Run on a small test dataset.")
    
    parser.add_argument(
        "-a", "--architecture",
        dest="architecture",
        default=False,
        choices=["one_layer", "experiment", "mnih", "two_layer", "one_layer_pool_2", "one_layer_pool_3", "one_layer_pool_4", "one_layer_filter_12",
                "one_layer_filter_16", "one_layer_filter_20", "two_layer_pool_2", "two_layer_pool_3", "two_layer_pool_4",
                "two_layer_filter_3", "two_layer_filter_4", "two_layer_filter_5", "two_layer_filter_6"],
        help="Neural net architecture.")
    
    parser.add_argument(
        "-v", "--visualise",
        dest="visualise",
        action="store_const",
        const=True,
        default=False,
        help="Visualise labels.")
    
    parser.add_argument(
        "-T", "--tensorboard",
        dest="tensorboard",
        action="store_const",
        const=True,
        default=False,
        help="Store tensorboard data while training.")
    
    parser.add_argument(
        "-C", "--checkpoints",
        dest="checkpoints",
        action="store_const",
        const=True,
        default=False,
        help="Create checkpoints while training.")
    
    parser.add_argument(
        "-E", "--earlystop",
        dest="earlystop",
        action="store_const",
        const=True,
        default=False,
        help="Create earlystop while training.")
    
    parser.add_argument(
        "--dataset",
        default="sentinel",
        choices=["sentinel"],
        help="Determine which dataset to use.")
    
    parser.add_argument(
        "--patch-size",
        default=64,
        type=int,
        help="Choose the patch size.")
    
    parser.add_argument(
        "--epochs",
        default=20,
        type=int,
        help="Number of training epochs.")
    
    parser.add_argument(
        "--distribution",
        default = 1.0,
        type=float,
        help="Percentage of building distribution in the image")
    
    parser.add_argument(
        "--model-id",
        default=None,
        type=str,
        help="Model that should be used. must be an existing ID.")
        
    parser.add_argument(
        "--setup",
        default=False,
        action="store_const",
        const=True,
        help="Create all necessary directories for the classifier to work.")
    
    parser.add_argument(
        "--out-format",
        default="GeoTIFF",
        choices=["GeoTIFF", "Shapefile"],
        help="Determine the format of the output for the evaluation method.")
    
    return parser



def main():
    parser = create_parser()
    args = parser.parse_args()
    
    if args.setup:
        create_directories()
        
    if args.debug:
        dataset = DATASETS['debug']
        args.dataset = "debug"
        features, _, labels, _ = preprocess_data(
            args.patch_size, args.distribution, dataset=dataset)
        #print(features, 'debug')
        #print("length of features: ",type(features), len(features),'element.shape: ',features[0][0])
        features_train, features_test = features[:100], features[100:120]
        labels_train, labels_test = labels[:100], labels[100:120]
    elif args.train_model or args.evaluate_model or args.preprocess_data:
        dataset = DATASETS[args.dataset]
        #print(dataset.values())
        load_from_cache = not args.preprocess_data
        try:
            features_train, features_test, labels_train, labels_test = preprocess_data(
                args.patch_size, args.distribution, dataset=dataset, only_cache=load_from_cache)
            #print(features_train, 'train_model or evaluate_model or preprocess_data')
            print("Length of features_train: ", len(features_train))
        except IOError:
                  print("Cache file does not exist. Please run again with -p flag.")
                  sys.exit(1)

        if args.visualise:
            visualise_labels(labels_train, args.patch_size, LABELS_DIR)
            visualise_labels(labels_test, args.patch_size, LABELS_DIR)
            
    if not args.model_id:
        timestamp = time.strftime("%d_%m_%Y_%H%M")
        model_id = "{}_{}_{}".format(timestamp, args.dataset, args.architecture)
    else:
        model_id = args.model_id
        
    if args.init_model or args.train_model or args.evaluate_model:
        model_dir = os.path.join(OUTPUT_DIR, model_id)
        save_makedirs(model_dir)
        
    
    # Hyperparameters for the model. Since there are so many of them it is 
    # more convenient to set them in the source code as opposed to passing 
    # them as arguments to the Command Line Interface. We use a list of tuples instead of a 
    # dict since we want to print the hyperparameters and for that purpose 
    # keep them in the predefined order.
    hyperparameters = [
        ("architecture", args.architecture),
        # Hyperparameters for the first convolutional layer.
        ("nb_filters_1", 64),
        ("filter_size_1", 9),
        ("stride_1", (2, 2)),
        # Hyperparameter for the first pooling layer.
        ("pool_size_1", (2, 2)),
        # Hyperparameter for the second convolutional layer (when 
        # two layer architecture is used).
        ("nb_filters_2", 128),
        ("filter_size_2", 5),
        ("stride_2", (1, 1)),
        # Hyperparameters for Stochastic Gradient Descent.
        ("learning_rate", 0.05),
        ("momentum", 0.9),
        ("decay", 0.0)
        ]
    
    hyperparameters_mnih = [
        ("architecture", args.architecture),
        # Hyperparameters for the first convolutional layer.
        ("nb_filters_1", 64),
        ("filter_size_1", 16),
        ("stride_1", (4, 4)),
        # Hyperparameter for the first pooling layer.
        ("pool_size_1", (2, 2)),
        ("pool_stride", 1),
        # Hyperparameter for the second convolutional layer).
        ("nb_filters_2", 112),
        ("filter_size_2", 4),
        ("stride_2", (1, 1)),
         # Hyperparameter for the third convolutional layer).
        ("nb_filters_3", 80),
        ("filter_size_3", 3),
        ("stride_3", (1, 1)),
     
        # Hyperparameters for Stochastic Gradient Descent.
        ("learning_rate", 0.05),
        ("momentum", 0.9),
        ("decay", 0.0)
        ]
    
    if args.init_model:
        model = init_model(args.patch_size, model_id, **dict(hyperparameters_mnih))
        save_model_summary(hyperparameters_mnih, model, model_dir)
    elif args.train_model or args.evaluate_model:
        hyperparameters = dict(hyperparameters_mnih)
        model = load_model(model_id)
        model = compile_model(model, hyperparameters["learning_rate"], hyperparameters['momentum'], hyperparameters["decay"])
        
    if args.train_model:
        model = train_model(
                model, features_train,
                labels_train, args.patch_size,
               model_id, model_dir, nb_epoch = args.epochs,
               checkpoints = args.checkpoints,
               tensorboard = args.tensorboard, earlystop = args.earlystop)
  
    if args.evaluate_model:
        evaluate_model(model, features_test, labels_test, args.patch_size, model_dir, out_format=args.out_format)

if __name__ == '__main__':
    main()
