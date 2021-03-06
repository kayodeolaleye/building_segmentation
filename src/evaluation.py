""" Evaluate a model's performance """

#import matplotlib
# Use 'Agg' backend to be able to use matplotlib in docker container.
#matplotlib.use('Agg')

import pickle
import os
import numpy as np
from sklearn import metrics
from geotiff_util import visualise_results
from model import get_matrix_form, normalise_input

import matplotlib.pyplot as plt
plt.style.use('ggplot')

def evaluate_model(model, features, labels, patch_size, out_path, out_format='GeoTIFF'):
    """ Calculate several metrics for the model and create a visualisation of the test dataset. """
    
    print('_' * 100)
    print('Start evaluating model.')
    
    X, y_true = get_matrix_form(features, labels, 16)
    X = normalise_input(X)
    #print("CHECK IF X IS NORMALIZED: ", X)
    #print('Model Metric Names: ', model.metrics_names)
    y_predicted = model.predict(X)
    #print("Y_TRUE: ", y_true)
    #print("Y_PREDICTED: ", y_predicted)
    predicted_bitmap = np.array(y_predicted)
    
    # Since the model only outputs probabilities for each pixel we have 
    # to transform them into 0s and 1s. For the sake of of simplicity we 
    # simply use a cut of value of 0.5.
    predicted_bitmap[0.5 <= predicted_bitmap] = 1
    predicted_bitmap[predicted_bitmap < 0.5] = 0
    
    false_positives = get_false_positives(predicted_bitmap, y_true)
    visualise_predictions(predicted_bitmap, labels, false_positives, patch_size, out_path, out_format=out_format)
    
    # We have to flatten our predictions and labels since by default the metrics are calculated by 
    # comparing the elements in the list of labels and predictions elementwise. So if we would not flatten
    # our results we would only get a true positive if we would predict every pixel in an entire patch right.
    # But we obviously only care about each pixel individually.
    y_true = y_true.flatten()
    y_predicted = y_predicted.flatten()
    predicted_bitmap = predicted_bitmap.flatten()
    
    print("Accuracy on test set: {}".format(metrics.accuracy_score(y_true, predicted_bitmap)))
    print("Precision on test set: {}".format(metrics.precision_score(y_true, predicted_bitmap)))
    print("Recall on test set: {}".format(metrics.recall_score(y_true, predicted_bitmap)))
    precision_recall_curve(y_true, y_predicted, out_path)

def visualise_predictions(predictions, labels, false_positives, patch_size, out_path, out_format="GeoTIFF"):
    """ Create a new GeoTIFF image which overlays the predictions of the model. """
    
    print("Create {} result files.".format(out_format))
    predictions = np.reshape(predictions,
                             (len(labels), patch_size, patch_size, 1))
    #print('predictions shape: ', predictions.shape)
    #print('predictions: ', predictions)
    false_positives = np.reshape(false_positives,
                                 (len(labels), patch_size, patch_size, 1))
    #print('False positives shape: ', false_positives.shape)
    #print('False positives: ', false_positives)
    results = []
    # We want to overlay the predictions and false positives on a GeoTIFF but we don't
    # have any information about the position in the source for each
    # patch in the predictions and false positives. We get this information from the labels.
    
    for i, (_, position, path_to_geotiff) in enumerate(labels):
        prediction_patch = predictions[i, :, :, :]
        false_positive_patch = false_positives[i, :, :, :]
        label_patch = labels[i][0]
        #print(path_to_geotiff)
        results.append(
            ((prediction_patch, label_patch, false_positive_patch), position, path_to_geotiff))
    #print("RESULTS: ", results)
    visualise_results(results, patch_size, out_path, out_format=out_format) 

def computeIoU(y_pred_batch, y_true_batch):
    return np.mean(np.asarray([pixelAccuracy(y_pred_batch[i], y_true_batch[i]) for i in range(len(y_true_batch))])) 
 
def pixelAccuracy(y_pred, y_true):
 #   y_pred = np.argmax(np.reshape(y_pred,[img_rows,img_cols]),axis=0)
    y_true = np.argmax(np.reshape(y_true,[img_rows,img_cols]),axis=0)
    y_pred = y_pred * (y_true>0)
     
    return 1.0 * np.sum((y_pred==y_true)*(y_true>0)) /  np.sum(y_true>0)

def precision_recall_curve(y_true, y_predicted, out_path):
    """ Create a PNG with the precision-recall curve for our predictions """
    
    print("Calculate precision recall curve.")
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_predicted)
    #print("y_true: {}, y_predicted: {}".format(y_true, y_predicted))
    #print("precision: {}, recall: {}, thresholds: {}".format(precision, recall, thresholds))
    # Save the raw precision and recall results to a pickle since we might want to
    # analyse them later
    out_file = os.path.join(out_path, "precision_recall.pickle")
    with open(out_file, "wb") as out:
        pickle.dump({
                "precision": precision,
                "recall": recall,
                "thresholds": thresholds
                }, out)
        
    # Create the precision-recall curve.
    out_file = os.path.join(out_path, "precision_recall.png")
    plt.clf()
    plt.plot(recall, precision, label="Precision-Recall curve")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.savefig(out_file)
        
def get_false_positives(predictions, labels):
    """ Get false positives for the given predicitions and labels. """
    
    FP = np.logical_and(predictions == 1, labels == 0)
    false_positives = np.copy(predictions)
    false_positives[FP] = 1
    false_positives[np.logical_not(FP)] = 0
    #print("FALSE POSITIVES: ", false_positives)
    return false_positives


"""def bb_IOU(boxA=[882.17,122.4,1241.,374.0]
                               , boxB=[867,117,1241,374]):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
 
    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)
 
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
 
    # return the intersection over union value
    return iou*100"""
 
