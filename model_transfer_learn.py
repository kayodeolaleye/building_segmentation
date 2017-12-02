""" Implementation of the convolutional neural net. """
import matplotlib
matplotlib.use("Pdf")
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, GlobalAveragePooling2D
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.constraints import maxnorm
from keras.layers import Dropout
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from folders import MODELS_DIR, TENSORBOARD_DIR, OUTPUT_DIR
from io_util import save_makedirs, save_model
print("keras: ", keras.__version__)
print("tensorflow: " ,tf.__version__)
np.random.seed(123)

def train_model(model, features, labels, patch_size, model_id, out_path,
                nb_epoch=10, checkpoints=False, tensorboard=False,
               earlystop=False):
    """ Train the model with the given features and labels """
    
    # The features and labels are a list of tuples when passed
    # to the function, Each tuple contains the patch and information
    # about its source image and its position in the source. To train 
    # the model we extract just the patches
    X, y = get_matrix_form(features, labels, patch_size)
    
    X = normalise_input(X)
    print('Shape of X: {}, Shape of y: {}'.format(X.shape, y.shape))
    # Directory which is used to store the model and its weights.
    model_dir = os.path.join(MODELS_DIR, model_id)
    
    earlystopper = None
    if earlystop:
        earlystop_file = os.path.join(model_dir, "early_weights.hdf5")
        #earlystopper = EarlyStopping(monitor='val_loss', patience=5)
        earlystopper = EarlyStopping(monitor='val_loss')

    checkpointer = None
    if checkpoints:
        checkpoints_file = os.path.join(model_dir, "weights.hdf5")
        checkpointer = ModelCheckpoint(checkpoints_file)
        
    tensorboarder = None
    if tensorboard:
        log_dir = os.path.join(TENSORBOARD_DIR, model_id)
        tensorboarder = TensorBoard(log_dir=log_dir)
        
    callbacks = [c for c in [earlystopper, checkpointer, tensorboarder] if c]
    
    print("Start training.")
    history = model.fit(X, y, epochs=nb_epoch, callbacks=callbacks, validation_split=0.25)#changed validation_split from 0.1 to 0.25
    plot_history(history, out_path)
    
    save_model(model, model_dir)
    return model 

def plot_history(history, out_path):
    print(history.history.keys())
    plot_accuracy(history, out_path)
    plot_loss(history, out_path)
    out_file = os.path.join(out_path, "history.pickle")
    with open(out_file, "wb") as out:
        pickle.dump({
                "acc": history.history['acc'],
                "val_acc": history.history['val_acc'],
                "loss": history.history['loss'],
                "val_loss": history.history['val_loss']
                }, out)
        
#  "Accuracy"
def plot_accuracy(history, out_path):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(os.path.join(out_path, 'history_acc.png'))
    
# "Loss"
def plot_loss(history, out_path):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.ylim(0, 0.9999)
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(os.path.join(out_path, 'history_loss.png'))

def fcn_32s(patch_size, num_channels):
    inputs = Input(shape=(patch_size, patch_size, num_channels))
    #print(inputs.shape)
    #import inception with pre-trained weights. do not include fully connected layers
    #inception_base = InceptionV3(weights='imagenet', include_top=False)
    resnet50_base = VGG16(weights='imagenet', include_top=False)
    
    # add a global spatial average pooling layer
    x = resnet50_base.output
    x = GlobalAveragePooling2D()(x)
    #x = Dropout(0.5)(x)
    #x = Conv2D(patch_size, (9, 9), subsample=(2,2), kernel_initializer='glorot_uniform')(x)
    #x = GlobalAveragePooling2D()(x)
    #x = Activation('relu')(x)
    #x = Conv2D(128, (5,5), subsample=(1,1))(x)
    #x = Dropout(0.5)(x)
    #x = Activation('relu')(x)
    # let's add a fully-connected layer
    x = Dense(512, activation='relu')(x)
    #x = Dropout(0.5)(x)
    # and a fully connected output/classification layer. 
    # CIFAR10 has 10 different classes (wowza)
    predictions = Dense(patch_size * patch_size, activation='sigmoid')(x)
    
    # create full model to train on 
    resnet50_transfer = Model(input=resnet50_base.input, output=predictions)
    
    # we want to train our new fully connected layers using the pre-trained weights
    # freeze all of the convolutional layers from Inceptionv3
    for layer in resnet50_base.layers:
        layer.trainable = False
    
    return resnet50_transfer

def init_model(patch_size, model_id, architecture='one_layer',
               nb_filters_1=64, filter_size_1=12,
               stride_1=(4, 4), pool_size_1=(3,3),
               nb_filters_2=128, filter_size_2=4,
               stride_2=(1,1), learning_rate=0.05,
               momentum=0.9, decay=0.0):
    """ Initialise a new model with the given hyperparameters and save it for later use. """
    
    num_channels = 3
    #model = Sequential()
    
    if architecture == 'one_layer':
        #-----------------------------------------------------------------
       
        model = fcn_32s(patch_size, num_channels)
        """model.add(Conv2D(patch_size, (9, 9), subsample=(2,2), input_shape=(patch_size, patch_size, num_channels), kernel_initializer='glorot_uniform'))
        #model.add(MaxPooling2D(pool_size=(2, 2), strides=1))
        model.add(Activation('relu'))
        model.add(Conv2D(128, (5,5), subsample=(1,1)))
        model.add(Activation('relu'))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(patch_size * patch_size))
        model.add(Activation('sigmoid'))"""
       
        #model.add(Flatten())
        #model.add(Dense(patch_size * patch_size))
        #model.add(Dense(patch_size * patch_size * 3))
        #model.add(Activation('sigmoid'))
        #use softmax with categorical cross entropy for multi-class classification task
        #-----------------------------------------------------------------------------------------
        
    elif architecture == 'two_layer':
        model.add(
            Conv2D(
                nb_filters_1, filter_size_1,
                filter_size_1, subsample=stride_1,
                input_shape=(patch_size, patch_size, num_channels)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=pool_size_1))
        model.add(
            Conv2D(
                nb_filters_2, filter_size_2,
                filter_size_2, subsample=stride_2))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(patch_size * patch_size))
        model.add(Activation('sigmoid'))
  
   
    model = compile_model(model, learning_rate, momentum, decay)
    
    # Print a summary of the model to the console.
    print("Summary of the model")
    model.summary()
    
    model_dir = os.path.join(MODELS_DIR, model_id)
    save_makedirs(model_dir)
    
    save_model(model, model_dir)
    
    return model
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#history = model.fit(x_test, y_test, nb_epoch=10, validation_split=0.2, shuffle=True)



def compile_model(model, learning_rate, momentum, decay):
    """ Compile the keras model with the given hyperparameters."""
    optimizer = SGD(lr=learning_rate, momentum=momentum, decay=decay, clipvalue=0.5)
    #optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, clipnorm=1.)
    """model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])"""
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    """model.compile(loss='jaccard_loss',
                  optimizer=optimizer,
                  metrics=['accuracy'])"""
    return model

def normalise_input(features):
    """ Normalise the features such that all values are in the range [0,1]. """
    features = features.astype(np.float32)
    
    return np.multiply(features, 1.0 / 255.0)

def get_matrix_form(features, labels, patch_size):
    """ Transform a list of tuples of features and labels to a matrix which contains
    only the patches used for training a model."""
    features = [patch for patch, position, path in features]
    labels =[patch for patch, position, path in labels]
    
    # The model will have one output corresponding to each pixel in the feature patch.
    # So we need to transform the labels which are given as a 2D bitmap into a vector.
    labels = np.reshape(labels, (len(labels), patch_size * patch_size))
    #labels = np.reshape(labels, (len(labels) * patch_size * patch_size,))
    return np.array(features), np.array(labels)
                
