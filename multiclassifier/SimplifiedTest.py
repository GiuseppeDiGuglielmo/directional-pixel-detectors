import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Standard library imports
import math
import h5py

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

# Fold BatchNormalization in QDense
def CreateQModel(shape, nb_classes):
    x = x_in = Input(shape, name="input1")    
    x = QDenseBatchnorm(58,
      kernel_quantizer=quantized_bits(4,0,alpha=1),
      bias_quantizer=quantized_bits(4,0,alpha=1),
      name="dense1")(x)    
    x = QActivation("quantized_relu(8,0)", name="relu1")(x)
    x = QDense(3,
        kernel_quantizer=quantized_bits(4,0,alpha=1),
        bias_quantizer=quantized_bits(4,0,alpha=1),
        name="dense2")(x)
    x = Activation("linear", name="linear")(x)
    model = Model(inputs=x_in, outputs=x)
    return model

# code to convert h5 file to csv
def h5ToCSV(h5_file_path):
    with h5py.File(h5_file_path, 'r') as f:
        for key in f.keys():
            print(f"layer: {key}")
            data = f[key][:]
            np.savetxt(f"{key}.csv", data, delimiter=",")

if __name__ == "__main__":

    # create model
    shape = 16 # y-profile ... why is this 16 and not 8?
    nb_classes = 3 # positive low pt, negative low pt, high pt
    first_dense = 58 # shape of first dense layer
    
    mtype = "qkeras"

    # keras
    if mtype == "keras":
        # initiate model
        model = CreateModel(shape, nb_classes, first_dense)
        model.summary()
        # load the model
        model_file = "/fasic_home/gdg/research/projects/CMS_PIX_28/directional-pixel-detectors/multiclassifier/models/ds8l6_padded_noscaling_keras_d58model.h5"
        model = tf.keras.models.load_model(model_file)

    # qkeras
    if mtype == "qkeras":
        # initiate model
        model = CreateQModel(shape, nb_classes)
        model.summary()
        # load the model
        model_file = "/fasic_home/gdg/research/projects/CMS_PIX_28/directional-pixel-detectors/multiclassifier/models/ds8l6_padded_noscaling_qkeras_foldbatchnorm_d58w4a8model.h5"
        co = {}
        utils._add_supported_quantized_objects(co)
        model = tf.keras.models.load_model(model_file, custom_objects=co)
        # Iterate through each layer and print weights and biases
        for layer in model.layers:
            print(f"Layer: {layer.name}")
            for weight in layer.weights:
                print(f"  {weight.name}: shape={weight.shape}")
                print(f"    Values:\n{weight.numpy()}\n")

    # load example inputs and outputs
    x_test = pd.read_csv("/asic/projects/C/CMS_PIX_28/benjamin/verilog/workarea/cms28_smartpix_verification/PnR_cms28_smartpix_verification_D/tb/dnn/csv/l6/input_1.csv", header=None)
    x_test = np.array(x_test.values.tolist())

    y_test = pd.read_csv("/asic/projects/C/CMS_PIX_28/benjamin/verilog/workarea/cms28_smartpix_verification/PnR_cms28_smartpix_verification_D/tb/dnn/csv/l6/layer7_out_ref_int.csv", header=None)
    y_test = np.array(y_test.values.tolist()).flatten()

    print(x_test.shape, y_test.shape)

    # get loss, accuracy
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test loss: {loss}")
    print(f"Test accuracy: {accuracy}")

    # make predictions
    predictions = model.predict(x_test)
    # print(predictions)
    predictions = np.argmax(predictions, axis=1)
    # print(predictions)

    # print some to screen
    #for x, y, p in zip(x_test, y_test, predictions):
    #    print("x, y, prediction: ", x, y, p)
