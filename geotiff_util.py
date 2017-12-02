""" Functions used for processing GeoTIFF Satellite images """

import rasterio
import rasterio.warp
import fiona
import os
import itertools
import numpy as np
from io_util import get_file_name
from folders import WGS84_DIR
import cv2
import skimage
from skimage.transform import rotate
def reproject_dataset(geotiff_path):
    """ Project a GeoTIFF to the WGS84 coordinate reference system."""
  
    out_path = geotiff_path
    
    return rasterio.open(out_path), out_path

        #return rasterio.open(out_path, 'w', **kwargs), out_path
    
"""def create_patches(bands_data, patch_size, path_to_geotiff):
    #Patch the satellite image which is given as a matrix into patches of the given size
    #bands_data = cv2.resize(bands_data, (1200,1200))
    rows, cols = bands_data.shape[0], bands_data.shape[1]
    #print('bands data: ', bands_data)
    all_patched_data = []
    
    # Cartesian projection of all the possible row and column indexes. This
    # gives all possible left-upper positions of our patches.
    patch_indexes = itertools.product(range(0, rows, patch_size), range(0, cols, patch_size))
    
    for (row, col) in patch_indexes:
        in_bounds = row + patch_size < rows and col + patch_size < cols #returns boolean
        if in_bounds:
            new_patch = bands_data[row:row + patch_size, col:col + patch_size]
            # In addition to the patch, also store its position, given by 
            # its upper left corner, and the path to the GeoTIFF it belongs
            # to. We need this information to visualise our results later on.
            all_patched_data.append((new_patch, (row, col), path_to_geotiff))
            
    return all_patched_data"""

def create_patches(feature_data, label_data, patch_size, path_to_geotiff, path_to_geotiff_bitmap, distribution):
    rows, cols = feature_data.shape[0], feature_data.shape[1]
    all_patched_feature = []
    all_patched_label = []
    # Cartesian projection of all the possible row and column indexes. This
    # gives all possible left-upper positions of our patches.
    #patch_indexes = itertools.product(range(0, rows, patch_size), range(0, cols, patch_size))
    for (row, col) in sliding_window(rows, cols, patch_size, step=8):
        in_bounds = row + patch_size < rows and col + patch_size < cols #returns boolean
        if in_bounds:
            new_patch_label = label_data[row:row + patch_size, col:col + patch_size] 
            #if new_patch_label.sum() * 100/np.dot(patch_size, patch_size) >= distribution:
            if new_patch_label.sum()>410: #205, #410, 820
                #new_patch_label = transform(new_patch_label, flip=True, mirror=True,rotations=[45])
                new_patch_feature = feature_data[row:row + patch_size, col:col + patch_size]
                #new_patch_feature = transform(new_patch_feature, flip=True, mirror=True,rotations=[45])
                #for transformed_label in new_patch_label:
                all_patched_label.append((new_patch_label, (row, col), path_to_geotiff_bitmap))
                #for transformed_feature in new_patch_feature:
                all_patched_feature.append((new_patch_feature, (row, col), path_to_geotiff))
    return all_patched_feature, all_patched_label

"""def create_patches(feature_data, label_data, patch_size, path_to_geotiff, path_to_geotiff_bitmap, distribution):
    rows, cols = feature_data.shape[0], feature_data.shape[1]
    all_patched_feature = []
    all_patched_label = []
    # Cartesian projection of all the possible row and column indexes. This
    # gives all possible left-upper positions of our patches.
    patch_indexes = itertools.product(range(0, rows, patch_size), range(0, cols, patch_size))
    for (row, col) in patch_indexes:
        in_bounds = row + patch_size < rows and col + patch_size < cols #returns boolean
        if in_bounds:
            new_patch_label = label_data[row:row + patch_size, col:col + patch_size] 
            if new_patch_label.sum() * 100/np.dot(patch_size, patch_size) >= distribution:
            #if new_patch_label.sum()>5:
                new_patch_feature = feature_data[row:row + patch_size, col:col + patch_size]
                all_patched_feature.append((new_patch_feature, (row, col), path_to_geotiff))
                all_patched_label.append((new_patch_label, (row, col), path_to_geotiff_bitmap))
    return all_patched_feature, all_patched_label"""

# Simple sliding window function
def sliding_window(nrow, ncol, patch_size,step=None):
    """Extract patches according to a sliding window.
    Args:            
        step (int, optional): The sliding window stride (defaults to 10px).
        patch_size(int, int, optional): The patch size (defaults to (20,20)).
        nrow, ncol: number of rows and columns in the tiff image
    Returns:
        list: list of patches with window_size dimensions
    """

    if step is None:
        step=patch_size/2 

    #print('step, patch_size',step, patch_size)

    # slide a window across the image
    for x in range(0, nrow, step):
        if x + patch_size > nrow:
            x = nrow - patch_size
        for y in range(0, ncol, step):
            if y + patch_size > ncol:
                y = ncol - patch_size
            yield x, y
            
def transform(patch,flip=True, mirror=True,rotations=[]):
    """Perform data augmentation on a patch.
    Args:
        patch (numpy array): The patch to be processed.
        flip (bool, optional): Up/down symetry.
        mirror (bool, optional): left/right symetry.
        rotations (int list, optional) : rotations to perform (angles in deg).
    Returns:
        array list: list of augmented patches
    """
    transformed_patches = [patch]
    for angle in rotations:
        transformed_patches.append(skimage.img_as_ubyte(rotate(patch, angle)))
    if flip:
        transformed_patches.append(np.flipud(patch))
    if mirror:
        transformed_patches.append(np.fliplr(patch))
    return transformed_patches

def image_from_patches(patches, patch_size, image_shape):
    """ 'Stitch' several patches back together to form one image. """
    
    image = np.zeros(image_shape, dtype=np.uint8)
    #image = cv2.resize(image, (1500, 1500))
    #print('image_from_patches shape: ', image.shape)
    for patch, (row, col), _ in patches:
        patch = np.reshape(patch, (patch_size, patch_size))
        image[row:row + patch_size, col:col + patch_size] = patch
    return image

def overlay_bitmap(bitmap, raster_dataset, out_path, color='blue'):
    """ Overlay the given satellite image with a bitmap """
    
    # RGB values for possible color options.
    colors = {
        "red": (255,0,0),
        "green": (0,255,0),
        "blue": (0,0,255)
        }
    image = raster_dataset.read()
    #print('overlay_bitmap image shape before resizing: ', image.shape)
    #image = np.rollaxis(image, 0, 3)
    #image = cv2.resize(image, (1500, 1500))
    #image = np.rollaxis(image, 2, 0)
    #print('overlay_bitmap image shape after resizing: ', image.shape) 
    red, green, blue = image
    #print('red: {}, green: {}, blue: {}'.format(red, green, blue))
    red[bitmap == 1] = colors[color][0]
    green[bitmap == 1] = colors[color][1]
    blue[bitmap == 1] = colors[color][2]
    
    profile = raster_dataset.profile
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(red, 1)
        dst.write(green, 2)
        dst.write(blue, 3)
        
    return rasterio.open(out_path)

def create_shapefile(bitmap, raster_dataset, out_path):
    
    shapes = rasterio.features.shapes(bitmap, transform=raster_dataset.transform)
    records = map(lambda geom, _: {"geometry": geom, "properties": {}}, shapes)
    schema = {
        "geometry": "Polygon",
        "properties": {}
        }
    
    with fiona.open(out_path, 'w', driver='ESRI Shapefile', crs=raster_dataset.crs, schema=schema) as f:
        f.writerecords(records)
        
def get_key(item):
  # print("item[2]", item[2])
    return item[2]

def visualise_labels(labels, patch_size, out_path):
    """ Given the labels of a satellite image as patches. Overlay the source image with the labels
    to check if labels are roughly correct."""
    
    # The patches might come from different satellite images so we have to 
    # group them according to their source image
    sorted_by_path = sorted(labels, key=get_key)
    for path, predictions in itertools.groupby(sorted_by_path, get_key):
        raster_dataset = rasterio.open(path)
        bitmap_shape = (raster_dataset.shape[0], raster_dataset.shape[1])
        #print('bitmap_shape: ', raster_dataset.shape)
        bitmap = image_from_patches(predictions, patch_size, bitmap_shape)
        #print('bitmap: ', bitmap)
        satellite_img_name = get_file_name(path)
        out_file_name = "{}.tif".format(satellite_img_name)
        out = os.path.join(out_path, out_file_name)
        overlay_bitmap(bitmap, raster_dataset, out)

def get_key_2(item):
    return item[0][0], item[1], item[2]

def get_key_3(item):           
    return item[0][1], item[1], item[2]

def get_key_4(item):              
    return item[0][2], item[1], item[2]

def visualise_results(results, patch_size, out_path, out_format="GeoTIFF"):
    """ Given the predictions, false positives and the labels of our model
    visualise them on the satellite image they belong to."""
    
    sorted_by_path = sorted(results, key=get_key)
    for path, result_patches in itertools.groupby(sorted_by_path, get_key):
    #   print("result_patches: ", result_patches)
   #    print("path: ", path)
        raster_dataset = rasterio.open(path)
        dataset_shape = (raster_dataset.shape[0], raster_dataset.shape[1])
        
        result_patches = list(result_patches)
        predictions = map(get_key_2, result_patches)
        labels = map(get_key_3, result_patches)
        false_positives = map(get_key_4, result_patches)

        satellite_img_name = get_file_name(path)
        file_extension = "tif" if out_format == "GeoTIFF" else "shp"
        out_file_name = "{}_results.{}".format(satellite_img_name, file_extension)
        out = os.path.join(out_path, out_file_name)
        
        if out_format == "GeoTIFF":
            # We first write the labels in blue, then predictions in green and then false positives in red.
            # This way, the true positives will be green, false positives red, false negatives blue and everything 
            # else in the image will be true negatives
            for patches, color in [(labels, 'blue'), (predictions, 'green'), (false_positives, 'red')]:
                bitmap = image_from_patches(patches, patch_size, dataset_shape)
                raster_dataset = overlay_bitmap(bitmap, raster_dataset, out, color=color)
        elif out_format == "Shapefile":
            bitmap = image_from_patches(predictions, patch_size, dataset_shape)
            create_shapefile(bitmap, raster_dataset, out)
    
