""" Transform GeoTIFF images and OSM shapefiles into feature and label matrices which can be used
to train the convNet """

import fiona
import pickle
import rasterio
from rasterio.features import rasterize
import rasterio.warp
import os
import sys
from folders import PATCHES_DIR, BUILDING_BITMAPS_DIR
from geotiff_util import reproject_dataset, create_patches
from io_util import get_file_name, save_patches, save_bitmap, load_bitmap
import numpy as np



def preprocess_data(patch_size, dist, dataset, only_cache=False):
    """ Create features and labels for a given dataset. The features are patches which contain
    the three RGB bands of the satellite image, so they have the form (patch_size, patch_size, 3).
    Labels are bitmaps with 1 indicating that the corresponding pixel in the satellite image 
    represents buildings."""
    
    print('_' * 100)
    print("Start preprocessing data.")
    
    features_train, labels_train = extract_features_and_labels(
        dataset["train"], patch_size, only_cache)
    features_test, labels_test = extract_features_and_labels(
        dataset["test"], patch_size, dist, only_cache)
    
    return features_train, features_test, labels_train, labels_test


def extract_features_and_labels(dataset, patch_size, dist, only_cache=False):
    """ For each satellite image and its corresponding shapefiles in the dataset create
    patched features and labels """
    features = []
    labels = []
    
    for geotiff_path, geotiff_bitmap_path in dataset:
        #print("geotiff_path: {}, geotiff_bitmap_path: {}".format(geotiff_path, geotiff_bitmap_path))
        patched_features, patched_labels = create_patched_features_and_labels(
            geotiff_path, geotiff_bitmap_path, patch_size, dist, only_cache)
        #print(patched_features)
        features += patched_features
        labels += patched_labels
        
    return features, labels

def create_patched_features_and_labels(geotiff_path, geotiff_bitmap_path, patch_size, dist, only_cache=False):
    """ Create the features and labels for a given satellite image and its shapefiles. """
    
    # Try to load patch from cache.
    satellite_img_name = get_file_name(geotiff_path)
    cache_file_name = "{}_{}.pickle".format(satellite_img_name, patch_size)
    cache_path = os.path.join(PATCHES_DIR, cache_file_name)
    try:
        print('Load patches from {}.'.format(cache_path))
        with open(cache_path, 'rb') as f:
            patches = pickle.load(f)
            
        return patches["features"], patches["labels"]
    except IOError as e:
        if only_cache:
            raise
        print("Cache not available. Compute patches.")
        
    # The provided satellite images have a different coordinate reference system as 
    # the familiar WGS84 which uses Latitude and Longitude. SO we need to reproject
    # the satellite image to the WGS84 coordinate reference system
    #print('geotiff_path: ',geotiff_path)
    dataset, wgs84_path = reproject_dataset(geotiff_path)
    bands = np.dstack(dataset.read())

    #print('bands,dataset,wgs84_path : ',bands,dataset,wgs84_path)    

    # For the given satellite image, create a bitmap which has 1 at every pixel corresponding
    # to building in the satellite image. In order to do this we use building polygons from OSM.
    #The building polygons are stored in forms of shapefiles and are given by "shapefiles_paths".
    
    building_bitmap = create_bitmap(geotiff_path, geotiff_bitmap_path, dataset)
    #patched_bands = create_patches(bands, patch_size, wgs84_path)
    #patched_bitmap = create_patches(building_bitmap, patch_size, wgs84_path)
    #patched_bitmap = create_patches(building_bitmap, patch_size, geotiff_bitmap_path)
    
    patched_bands, patched_bitmap = create_patches(bands, building_bitmap, patch_size, wgs84_path, geotiff_bitmap_path, dist)
    # Due to the projection, the satellite image in the GeoTIFF is not a perfect rectangle and the 
    # remaining space on the edges is blacked out. When we overlay the GeoTIFF with the 
    # shapefile it also overlays features for the blacked out parts which means that if we don't 
    # remove these patches the classifier will be fed with non-empty labels for empty features.
    
    #patched_bands, patched_bitmap = remove_edge_patches(patched_bands, 
     #                                             patched_bitmap, patch_size, dataset.shape)
    
    save_patches(cache_path, patched_bands, patched_bitmap)
    
    return patched_bands, patched_bitmap


def remove_edge_patches(patched_bands, patched_bitmap, patch_size, source_shape):
    """ Remove patches which are on the edge of the satellite image and which contain blacked out
    content"""
    
    EDGE_BUFFER = 350
    
    rows, cols = source_shape[0], source_shape[1]
    
    bands = []
    bitmap = []
    #print('***** len(patched_bands)',len(patched_bands),type(patched_bands))
    #print('patched_bands_0',patched_bands[0])
    for i, (patch, (row, col), _) in enumerate(patched_bands):
        is_in_center = EDGE_BUFFER <= row and row <= (
            rows-EDGE_BUFFER) and EDGE_BUFFER <= col and col <= (
            cols-EDGE_BUFFER)
        # Checks whether our patch contains a pixel which is only black.
        # This might also delete patches which contain a natural feature which is 
        # totally black but these are only a small number of patches and we don't 
        # care about deleting them as well.
        contains_black_pixel = [0, 0, 0] in patch
        is_edge_patch = contains_black_pixel and not is_in_center
        #print('i, row, col',i, row, col,is_in_center,contains_black_pixel,is_edge_patch)
        #print('patch',patch)
        if not is_edge_patch:
            bands.append(patched_bands[i])
            bitmap.append(patched_bitmap[i])
          
    return bands, bitmap


def create_bitmap(satellite_path, satellite_bitmap_path,  raster_dataset):
    """ load the bitmap for a given satellite image. """
    #print('satellite_path: {}, satellite_bitmap_path: {}'.format(satellite_path, satellite_bitmap_path))
    satellite_img_name = get_file_name(satellite_path)
    cache_file_name = "{}_building.tif".format(satellite_img_name)
    cache_path = os.path.join(BUILDING_BITMAPS_DIR, cache_file_name)
    try:
        # Try loading the building bitmap from cache.
        print("Load building bitmap {}".format(cache_path))
        bitmap = load_bitmap(cache_path)
        bitmap[bitmap ==255] = 1
        return bitmap
    except IOError as e:
        print("No cache file found.")
        
    print('Create bitmaps for building features')
    print("Load shapefile {}.".format(satellite_bitmap_path))
    bitmap_image = rasterio.open(satellite_bitmap_path).read(1)
    
    save_bitmap(cache_path, bitmap_image, raster_dataset)
    
    bitmap_image[bitmap_image == 255] = 1
    return bitmap_image