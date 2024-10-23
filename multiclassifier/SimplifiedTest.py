import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Standard library imports
import math

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from pandas import read_csv
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

# TensorFlow imports
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, Input, Lambda, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# QKeras imports
from qkeras import *

# Local imports
from hep_utils import plot_pt_filter, get_number_of_tracks, get_bkg_rej
from utils import *

# number with dense
def CreateModel(shape, nb_classes, first_dense):
    x = x_in = Input(shape, name="input")
    x = Dense(first_dense, name="dense1")(x)
    x = keras.layers.BatchNormalization()(x)
    x = Activation("relu", name="relu1")(x)
    x = Dense(nb_classes, name="dense2")(x)
    x = Activation("linear", name="linear")(x)
    model = Model(inputs=x_in, outputs=x)
    return model

# convert 8 number y-profile into 16 bits by quantizing each value
def quantizeInputs(x):
    
    # define mapping for encoding into 2 bit inputs
    # X_train: max= 25 , log2(max)= 5 , min= 0
    # X_test: max= 21 , log2(max)= 5 , min= 0
    max_value_X_train = 25 # np.max(X_train)
    min_value_X_train = 0 # np.min(X_train)
    max_value_X_test = 21 # np.max(X_test)
    min_value_X_test = 0 # np.min(X_test)
    log2_max_value_X_train = int(np.ceil(math.log2(np.abs(max_value_X_train))))
    log2_max_value_X_test = int(np.ceil(math.log2(np.abs(max_value_X_test))))
    print('X_train: max=', max_value_X_train, ', log2(max)=', log2_max_value_X_train, ', min=', min_value_X_train)
    print('X_test: max=', max_value_X_test, ', log2(max)=', log2_max_value_X_test, ', min=', min_value_X_test)

    # set threshold values
    threshold1 = 10  
    threshold2 = 20 

    # update with logic for how to convert y-profile of 8 numbers into 16 numbers
    x_quant = []
    for arr in x:
        x_quant.append([])
        for i in arr:
             bit1 = 1 if i > threshold1 else 0
             bit2 = 1 if i > threshold2 else 0
             x_quant[-1].append(bit1)
             x_quant[-1].append(bit2)

    # convert to np array
    x_quant = np.array(x_quant)

    return x_quant

if __name__ == "__main__":
    
    # create model
    shape = 16 # y-profile ... why is this 16 and not 8?
    nb_classes = 3 # positive low pt, negative low pt, high pt
    first_dense = 58 # shape of first dense layer
    model = CreateModel(shape, nb_classes, first_dense)
    model.summary()

    # load the model
    model_file = "/fasic_home/gdg/research/projects/CMS_PIX_28/directional-pixel-detectors/multiclassifier/models/ds8l0_padded_noscaling_keras_d58model.h5"
    # co = {}
    # utils._add_supported_quantized_objects(co)
    model = tf.keras.models.load_model(model_file) #, custom_objects=co)
    
    # define some test input data
    x_test = [
        [0, 0, 5, 15, 15, 7, 0, 0], # row 7-0 of an example y-profile that is 8 numbers long because we have a 8x32
    ]
    y_test = [2]
    
    # convert to np and print shapes
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
    # quantize -> NEEDS TO BE UPDATED
    x_test = quantizeInputs(x_test)
    print(x_test)
    print(y_test)

    # get loss, accuracy
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test loss: {loss}")
    print(f"Test accuracy: {accuracy}")

    # make predictions
    predictions = model.predict(x_test) 
    print(predictions)
    predictions = np.argmax(predictions, axis=1)
    print(predictions)
