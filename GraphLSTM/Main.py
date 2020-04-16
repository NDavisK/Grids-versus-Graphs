import torch.utils.data as utils
import torch
import numpy as np
import pandas as pd
from Models import * 
from Train_Validate import * 
from torch.autograd import Variable
import time
from Models import *
import math
import sklearn
from math import sqrt
from sklearn.metrics import mean_squared_error
import pickle
from pandas import read_csv

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
    


def PrepareDataset(taxi_demand_matrix, BATCH_SIZE = 24, seq_len = 10, pred_len = 1, train_propotion = 0.7, valid_propotion = 0.2):
    """ Prepare training and testing datasets and dataloaders.
    
    Convert taxi_demand matrix to training and testing dataset. 
    The vertical axis of taxi_demand_matrix is the time axis and the horizontal axis 
    is the spatial axis.
    
    Args:
        taxi_demand_matrix: a Matrix containing spatial-temporal taxi_demand data for a network
        seq_len: length of input sequence
        pred_len: length of predicted sequence
    Returns:
        Training dataloader
        Testing dataloader
    """
    time_len = taxi_demand_matrix.shape[0]
    
    max_taxi_demand = taxi_demand_matrix.max().max()
    
    taxi_demand_matrix =  taxi_demand_matrix / max_taxi_demand
    
    taxi_demand_sequences, taxi_demand_labels = [], []
    
    for i in range(time_len - seq_len - pred_len):
        taxi_demand_sequences.append(taxi_demand_matrix.iloc[i:i+seq_len].values)
        
        taxi_demand_labels.append(taxi_demand_matrix.iloc[i+seq_len:i+seq_len+pred_len].values)
    
    taxi_demand_sequences, taxi_demand_labels = np.asarray(taxi_demand_sequences), np.asarray(taxi_demand_labels)
    
    # shuffle and split the dataset to training and testing datasets
    sample_size = taxi_demand_sequences.shape[0]
    
    index = np.arange(sample_size, dtype = int)
    
    train_index = int(np.floor(sample_size * train_propotion))
    
    valid_index = int(np.floor(sample_size * ( train_propotion + valid_propotion)))
    
    train_data, train_label = taxi_demand_sequences[:train_index], taxi_demand_labels[:train_index]
    
    valid_data, valid_label = taxi_demand_sequences[train_index:valid_index], taxi_demand_labels[train_index:valid_index]
    
    test_data, test_label = taxi_demand_sequences[valid_index:], taxi_demand_labels[valid_index:]
    
    train_data, train_label = torch.Tensor(train_data), torch.Tensor(train_label)
    
    valid_data, valid_label = torch.Tensor(valid_data), torch.Tensor(valid_label)
    
    test_data, test_label = torch.Tensor(test_data), torch.Tensor(test_label)
    
    print("train_data.shape:", train_data.shape)
    
    print("train_labels.shape:", train_label.shape)
    
    print("valid_data.shape:", valid_data.shape)
    
    print("test_data.shape:", test_data.shape)
    
    train_dataset = utils.TensorDataset(train_data, train_label)
    
    valid_dataset = utils.TensorDataset(valid_data, valid_label)
    
    test_dataset = utils.TensorDataset(test_data, test_label)
    
    train_dataloader = utils.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True, drop_last = True)
    
    valid_dataloader = utils.DataLoader(valid_dataset, batch_size = BATCH_SIZE, shuffle=True, drop_last = True)
    
    test_dataloader = utils.DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=True, drop_last = True)
    
    return train_data, test_data, train_dataloader, valid_dataloader, test_dataloader, max_taxi_demand
    
#load data
taxi_demand_matrix =  pd.read_pickle('demand_geohash_60mins.obj') #taxi demand data 

A = np.load('demand_geohash.npy') #Adjacency matrix

A = A+(np.eye(np.shape(A)[0], dtype = int))

train_data, test_data, train_dataloader, valid_dataloader, test_dataloader, max_taxi_demand = PrepareDataset(taxi_demand_matrix)

losses_mase = []

losses_smape = []

losses_rmse = []

horizon = 24

for rerunloop in range(5):
    gclstm, gclstm_loss = TrainGraphLSTM(train_dataloader, valid_dataloader, A, K=1, back_length = 2, num_epochs = 500, Clamp_A = True)
    
    #Testing..
    inputs, labels = next(iter(test_dataloader))
    
    [batch_size, step_size, fea_size] = inputs.size()
    
    cur_time = time.time()
    
    pre_time = time.time()
    
    use_gpu = torch.cuda.is_available()
    
    loss_MSE = torch.nn.MSELoss()
    
    loss_L1 = torch.nn.L1Loss()
    
    tested_batch = 0
    
    losses_mse = []
    
    losses_l1 = [] 
    
    mase_perloader = []
    
    smape_perloader = []
    
    rmse_perloader = []
    
    for data in test_dataloader:
        
        inputs, labels = data
        
        if inputs.shape[0] != batch_size:
            continue
    
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else: 
            inputs, labels = Variable(inputs), Variable(labels)

        Hidden_State, Cell_State = gclstm.loop(inputs)
        
        loss_MSE = torch.nn.MSELoss()
        
        loss_L1 = torch.nn.L1Loss()
        
        loss_mse = loss_MSE(Hidden_State, labels)
        
        loss_l1 = loss_L1(Hidden_State, labels)
        
        losses_mse.append(loss_mse.data)
        
        losses_l1.append(loss_l1.data)
        

        Hidden_state = max_taxi_demand*Hidden_State.data.cpu() 
        
        Labels = max_taxi_demand*labels.data.cpu()
        
        Train_data = max_taxi_demand*train_data
        
        for counter in range(np.shape(Hidden_State)[1]):
            
            mase_perloader.append(MASE(train_data[:,9,counter],Labels[:horizon,0,counter],Hidden_state[:horizon,counter]))
            
            smape_perloader.append(SMAPE(Labels[:horizon,0,counter],Hidden_state[:horizon,counter]))
            
            rmse_perloader.append(RMSE(Labels[:horizon,0,counter],Hidden_state[:horizon,counter]))
    
    losses_mase.append(mase_perloader)
    
    losses_smape.append(smape_perloader)
    
    losses_rmse.append(rmse_perloader)


losses_mase = np.array(losses_mase)/max_taxi_demand

losses_smape = np.array(losses_smape)

losses_rmse = np.array(losses_rmse)

print('Tested: MASE_mean: {}, MASE_std : {}'.format(np.mean(losses_mase), np.std(losses_mase)))

print('Tested: SMAPE_mean: {}, SMAPE_std : {}'.format(np.mean(losses_smape), np.std(losses_smape)))

print('Tested: RMSE_mean: {}, RMSE_std : {}'.format(np.mean(losses_rmse), np.std(losses_rmse)))
           
            
        
