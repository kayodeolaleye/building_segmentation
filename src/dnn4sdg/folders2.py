""" This sets up the directories where the program can find the data """
#lustre onitemi data mbd training

import os
from os.path import join

# Specify the base directory for the data paths
#BASE_DIR = join(os.environ["HOME"], "THIS_COMPUTER", "Disk_Space", "kayode", "water_shapes", "data")
BASE_DIR_INPUT = join(os.environ["HOME"], "lustre", "onitemi", "data")
BASE_DIR = join(os.environ["HOME"], "lustre", "onitemi", "data")
print('base dir: ',BASE_DIR)
#Directories for the satellite images
SENTINEL_TRAIN_DIR = join(BASE_DIR_INPUT, "mbd", "training")
SHAPEFILE_TRAIN_DIR = join(BASE_DIR_INPUT, "mbd", "shapefiles")
SENTINEL_TEST_DIR = join(BASE_DIR_INPUT, "mbd", "test")
SHAPEFILE_TEST_DIR = join(BASE_DIR_INPUT, "mbd", "shapefiles")

#Directories to store everything related to the training data.
TRAIN_DATA_DIR = join(BASE_DIR_INPUT, "working", "train_data")
PATCHES_DIR = join(TRAIN_DATA_DIR, "patches")
print('cache dir: ', PATCHES_DIR)

BUILDING_BITMAPS_DIR = join(TRAIN_DATA_DIR, "building_bitmaps")
print('cache dir: ', BUILDING_BITMAPS_DIR)
WGS84_DIR = join(TRAIN_DATA_DIR, "WGS84_images")
LABELS_DIR = join(TRAIN_DATA_DIR, "labels_images")

#Directories to store the models and weights
MODELS_DIR = join(BASE_DIR_INPUT, "working", "models")
print('models dir: ', MODELS_DIR)

#Directories for model output (models, visualisations, etc)
OUTPUT_DIR = join(BASE_DIR, "output")
TENSORBOARD_DIR = join(OUTPUT_DIR, "tensorboard")

#===================================================================================================
MUENSTER_SHAPEFILE = join(SHAPEFILE_TRAIN_DIR,"mb.shp")                 
NETHERLANDS_SHAPEFILE = join(SHAPEFILE_TRAIN_DIR, "mb.shp")
MUENSTER_SHAPEFILE = join(SHAPEFILE_TRAIN_DIR, "mb.shp")
#TODO: Create or Search for shapefiles for shanties and other small building structures.
MUNICH_SHAPEFILE = join(SHAPEFILE_TRAIN_DIR, "mb.shp")
LIVERPOOL_SHAPEFILE = join(SHAPEFILE_TRAIN_DIR, "mb.shp")
BUDAPEST_SHAPEFILE = join(SHAPEFILE_TEST_DIR, "mb.shp")
#MELROSE_SHAPEFILE = join(SHAPEFILE_DIR, "mb.shp")
VENICE_SHAPEFILE = join(SHAPEFILE_TRAIN_DIR, "mb.shp")
#====================================================================================================
# Paths to satellite images

MUENSTER_SATELLITE =join(SENTINEL_TRAIN_DIR, "22679020_15.tiff")
NETHERLANDS_SATELLITE = join(SENTINEL_TRAIN_DIR, "22678930_15.tiff")
MUNICH_SATELLITE = join(SENTINEL_TRAIN_DIR, "22678945_15.tiff")
LIVERPOOL_SATELLITE = join(SENTINEL_TRAIN_DIR,"22678960_15.tiff")
LONDON_SATELLITE = join(SENTINEL_TRAIN_DIR,"22678975_15.tiff")
BUDAPEST_SATELLITE = join(SENTINEL_TEST_DIR, "22828930_15.tiff")
#MELROSE_SATELLITE = join(SENTINEL_DIR, "22678990_15.tiff")
VENICE_SATELLITE = join(SENTINEL_TRAIN_DIR, "22679005_15.tiff")
#=====================================================================================================


# Map shapefiles to the corresponding satellite image
SENTINEL_DATASET_TRAIN = [(NETHERLANDS_SATELLITE, [NETHERLANDS_SHAPEFILE]),
                          (MUENSTER_SATELLITE, [MUENSTER_SHAPEFILE]),
                          (MUNICH_SATELLITE, [MUNICH_SHAPEFILE]),
                          (LIVERPOOL_SATELLITE, [LIVERPOOL_SHAPEFILE]),
                          (VENICE_SATELLITE, [VENICE_SHAPEFILE])
                          ]
SENTINEL_DATASET_TEST = [(BUDAPEST_SATELLITE, [BUDAPEST_SHAPEFILE])]
#SENTINEL_DATASET_TEST = [(MELROSE_SATELLITE, [MELROSE_SHAPEFILE])]
                         
SENTINEL_DATASET = {
    "train": SENTINEL_DATASET_TRAIN,
    "test": SENTINEL_DATASET_TEST
    }

# Small dataset used for debugging purposes
DEBUG_DATASET = {
    "train": [(MUENSTER_SATELLITE, MUENSTER_SHAPEFILE)],
    "test": []
    }

DATASETS = {"sentinel":SENTINEL_DATASET, "debug": DEBUG_DATASET}
                          
