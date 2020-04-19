import numpy as np
from numpy.random import seed
from numpy import concatenate
seed(1)
import pandas
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import math
from math import sqrt
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from matplotlib.backends.backend_pdf import PdfPages
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import *
from tensorflow.python.keras.optimizers import *
from tensorflow.python.keras.callbacks import EarlyStopping
stop = EarlyStopping(monitor='val_loss',min_delta = 0.000001,patience=60,verbose = 0)


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
    
    errors = np.abs(testing_series - prediction_series)
    
    return errors.mean()/d

def RMSE(y_actual,y_predicted):
    
    rms = sqrt(mean_squared_error(y_actual, y_predicted))
    
    return rms

import keras.backend as K

def WeightedLoss1(y_true,y_pred):
     
     wsum = (K.mean(K.square(y_pred - y_true), axis=-1))+ (K.mean(K.abs(y_pred - y_true), axis=-1))
    
    return wsum
    
    

dataset = read_csv("demand_geohash.csv", header=0) #data dimension 740*1440*9 740 examples, 1440 sequence length, 9 features

dim3 = no_features # =9

dim1 = no_examples # =740

dataset.drop(dataset.columns[[0]],axis =1 ,inplace = True)

input2LSTM = np.zeros((dim1,np.shape(dataset)[0]-1,dim3+1))

loop = 0

forecast_horizon = 24

for counter in range(0,(np.shape(dataset)[1]),dim3):
    
    loopset = dataset.iloc[:,counter:counter+dim3] 
    
    values = loopset.values
    
    values = values.astype('float32')
    
    df = pandas.DataFrame(values)
    
    reframed = series_to_supervised(df, 1, 1)
    
    reframed1 = reframed.iloc[:,0:(np.shape(loopset)[1]+1)]
    
    input2LSTM[loop,:] = reframed1.values
    
    loop = loop+1
    
input2LSTM = np.concatenate((input2LSTM, input2LSTM[:(128-dim1%128),:,:]),axis = 0)

'''
Train LSTM models with varying number of features: 
fit an LSTM model with the top correlated feature, then with two top features, .. , 
in the end, compare and find the model with optimal #features

Suitable hyper parameters obtained by hyperopt
'''


for sicounter in range(8):
    
    r = range(sicounter+1)
    
    r.append(-1)
    
    input2LSTM1 = input2LSTM[:,:,r]
    
    dim3 = sicounter+1

    input2LSTMt = input2LSTM1.transpose([1,0,2])
    
    n_train_hours = np.shape(input2LSTMt)[0]-forecast_horizon
    
    train,test = input2LSTMt[:n_train_hours,:, :], input2LSTMt[n_train_hours:,:, :]
    
    train_2d = train.reshape(np.shape(train)[0],np.shape(train)[1]*np.shape(train)[2])
    
    test_2d = test.reshape(np.shape(test)[0],np.shape(test)[1]*np.shape(test)[2]) 

    scaler_train = MinMaxScaler(feature_range=(0, 1)).fit(train_2d)
    
    scaled_train = scaler_train.transform(train_2d)
    
    scaled_test = scaler_train.transform(test_2d)

    test_3d_t = scaled_test.reshape(np.shape(test)[0],np.shape(test)[1],np.shape(test)[2])
    
    train_3d_t = scaled_train.reshape(np.shape(train)[0],np.shape(train)[1],np.shape(train)[2])
    
    test_3d = test_3d_t.transpose([1,0,2])
    
    train_3d = train_3d_t.transpose([1,0,2])

    train_X_3d, train_y_3d = train_3d[:,:, :-1], train_3d[:,:, -1].reshape(np.shape(train_3d)[0], np.shape(train_3d)[1],1)
    
    test_X_3d, test_y_3d = test_3d[:,:, :-1], test_3d[:,:, -1].reshape(np.shape(test_3d)[0], np.shape(test_3d)[1],1)
    
    hcounter = 0
    
    training_epochs = 500
    
    b_s = 64
    
    L1 = [20]
    
    L2 = [0]
    
    d_1 = 0.368
    
    maselist = []
    
    smapelist = []
    
    rmselist = []
    
    #models are rerun 5 times and results are averaged to get mean(errors) +- standard deviation
    
    for rerunloop in range(5):
        
        model = Sequential()
        
        model.add(LSTM(L1[hcounter],return_sequences=True,input_shape=(None,dim3)))
        
        model.add(Dropout(d_1))
        
        model.add(Dense(1,activation = 'linear'))
        
        adam = Adam(lr= 10**-1) 
        
        model.compile(loss='WeightedLoss1', optimizer=adam)
        
        model.fit(train_X_3d,train_y_3d,batch_size =b_s, validation_split = 0.1,epochs=training_epochs,callbacks = [stop],verbose=0,shuffle = False) #-------SAVE MODEL
        
        X_3d = np.zeros((np.shape(input2LSTM1)[0],np.shape(input2LSTM1)[1],np.shape(input2LSTM1)[2]-1))
        
        X_3d[:,:n_train_hours,:] = train_X_3d
        
        X_3d[:,n_train_hours:,:] = test_X_3d
        
        selfpredictions = model.predict(X_3d)
        
        if (np.isnan(selfpredictions).any() == True):
            continue
            
        selfpredictions = selfpredictions[:,-np.shape(test_y_3d)[1]:,:] 
        
        to_transpose = np.concatenate((test_X_3d,selfpredictions), axis = 2)
        
        transposed = to_transpose.transpose([1,0,2])
        
        to_invert = transposed.reshape(np.shape(transposed)[0], np.shape(transposed)[1]*np.shape(transposed)[2])
        
        inverted = scaler_train.inverse_transform(to_invert)
        
        predictions_3d = inverted.reshape(np.shape(test)[0], np.shape(test)[1], np.shape(test)[2])
        
        predictions_3d = predictions_3d.transpose([1,0,2])
        
        predictions = predictions_3d[:,:,-1]
      
        test_y = input2LSTM1[:,n_train_hours:,-1]
        
        mase = []
        
        smape = []
        
        rmse = []
        
        for counter in range(np.shape(test_X_3d)[0]):
            
            mase.append(MASE(input2LSTM1[counter,:n_train_hours,0],test_y[counter,:],fromfullpred_actuals[counter,:]))
            
            smape.append(SMAPE(test_y[counter,:],fromfullpred_actuals[counter,:]))
            
            rmse.append(RMSE(test_y[counter,:],fromfullpred_actuals[counter,:]))
            
        
        maselist.append(mase)
        
        smapelist.append(smape)
        
        rmselist.append(rmse)

   
