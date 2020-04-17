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

def WeightedLoss(y_true,y_pred):
    
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true+y_pred+1),K.epsilon(),None))
    
    wsum = (K.mean(K.square(y_pred - y_true), axis=-1))+ (K.mean(K.abs(y_pred - y_true), axis=-1))
    
    return wsum
    
train_mean = get_train_mean() # a function that returns mean of training data

train_sd = get_train_sd() # a function that returns std deviation of training data

frame_height_width = 3 # we use 3*3 frames

no_examples = 740

sequence_length = 1439

X_frames = np.zeros((no_examples,sequence_length,frame_height_width,frame_height_width,1))

Y_frames = np.zeros((no_examples,sequence_length,frame_height_width,frame_height_width,1))

for outer_counter in range(no_examples):
    
    loop = 0
    
    filename = "training_example_"+str(outer_counter+1)+".csv" #each csv file containes sequence_length number of 3*3 frames
    
    dataset = read_csv(filename,header = 0)
    
    dataset.drop(dataset.columns[[0]],axis =1 ,inplace = True)
    
    values = dataset.values
    
    values = (values - train_mean)/train_sd
    
    values = values.astype('float32')
    
    for counter in range(0,(np.shape(dataset)[0])-frame_height_width,frame_height_width):
        
        X_frames[outer_counter,loop,:,:,0] = values[counter:counter+frame_height_width,:]
        
        Y_frames[outer_counter,loop,:,:,0] = values[(counter+frame_height_width):(counter+(2*frame_height_width)),:]
        
        loop = loop+1    
        
        
forecast_horizon = 24

n_train_hours = np.shape(X_frames)[1]-forecast_horizon

train_x,train_y = X_frames[:,:n_train_hours,:,:,:], Y_frames[:,:n_train_hours,:,:,:]

test_x,test_y = X_frames[:,n_train_hours:,:,:,:], Y_frames[:,n_train_hours:,:,:,:]

#use hyperopt to arrive at the optimal set of hyperparameters
seq = Sequential()

seq.add(ConvLSTM2D(filters=16, kernel_size=(3, 3),
                   input_shape=(None, frame_height_width, frame_height_width, 1),
                   padding='same', return_sequences=True))

seq.add(BatchNormalization())

seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
               activation='linear',
               padding='same', data_format='channels_last'))

seq.compile(loss=WeightedLoss, optimizer='adam')


maselist = []

smapelist = []

rmselist = []

for rerunloop in range(5):
    
    history = seq.fit(train_x, train_y, batch_size=64, epochs=500, validation_split=0.1, verbose = 1, callbacks = [stop])  
    
    pyplot.plot(history.history['loss'], label='train')
    
    pyplot.plot(history.history['val_loss'], label='test')
    
    pyplot.legend()
    
    pyplot.show()
    
    predictions = seq.predict(test_x)

    mase_perrun = []
    
    smape_perrun = []
    
    rmse_perrun = []
    
    for counter in range(np.shape(X_frames)[0]):
        
        y_hat = predictions[counter,:,1,1,0] 
        
        y = test_y[counter,:,1,1,0]
        
        y_train = train_x[counter,:,1,1,0]
        
        mase_perrun.append(MASE(y_train,y,y_hat))
        
        smape_perrun.append(SMAPE(y,y_hat))
        
        rmse_perrun.append(RMSE(y,y_hat))

    maselist.append(mase_perrun)
    
    smapelist.append(smape_perrun)
    
    rmselist.append(rmse_perrun)
    


