import numpy as np
from numpy.random import seed
from numpy import concatenate
seed(1)
import pandas
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import math
import tensorflow as tf
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from matplotlib.backends.backend_pdf import PdfPages
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import *
from tensorflow.python.keras.optimizers import *
from tensorflow.python.keras.layers.convolutional import Conv3D
from tensorflow.python.keras.layers.convolutional_recurrent import ConvLSTM2D
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.callbacks import EarlyStopping
stop = EarlyStopping(monitor='val_loss',min_delta = 0.001,patience=60,verbose = 0)


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t - %d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def SMAPE(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + y_pred +1))) * 100

def MASE(training_series, testing_series, prediction_series):
    n = training_series.shape[0]
    d = np.abs(  np.diff( training_series) ).sum()/(n-1)
    errors = np.abs(testing_series - prediction_series )
    return errors.mean()/d

def RMSE(y_actual,y_predicted):
    rms = sqrt(mean_squared_error(y_actual, y_predicted))
    return rms

import keras.backend as K

def WeightedLoss1(y_true,y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true+y_pred+1),K.epsilon(),None))
    wsum = (K.mean(K.square(y_pred - y_true), axis=-1))+ (K.mean(K.abs(y_pred - y_true), axis=-1))#+ (100.*K.mean(diff, axis=-1))
    return wsum
    
    
