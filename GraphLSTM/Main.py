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
    
    Convert taxi_demand/volume/occupancy matrix to training and testing dataset. 
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
    print(time_len)
    max_taxi_demand = taxi_demand_matrix.max().max()
    taxi_demand_matrix =  taxi_demand_matrix / max_taxi_demand
    
    taxi_demand_sequences, taxi_demand_labels = [], []
    for i in range(time_len - seq_len - pred_len):
        taxi_demand_sequences.append(taxi_demand_matrix.iloc[i:i+seq_len].values)
        taxi_demand_labels.append(taxi_demand_matrix.iloc[i+seq_len:i+seq_len+pred_len].values)
    taxi_demand_sequences, taxi_demand_labels = np.asarray(taxi_demand_sequences), np.asarray(taxi_demand_labels)
    print(taxi_demand_sequences.shape)
    # shuffle and split the dataset to training and testing datasets
    sample_size = taxi_demand_sequences.shape[0]
    index = np.arange(sample_size, dtype = int)
    np.random.shuffle(index)
    
    train_index = 1300 #int(np.floor(sample_size * train_propotion))
    valid_index = 1405 #int(np.floor(sample_size * ( train_propotion + valid_propotion)))
    
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
    

class GraphConvolutionalLSTM(nn.Module):
    
    def __init__(self, K, A, feature_size, Clamp_A=True):
        '''
        Args:
            K: K-hop graph
            A: adjacency matrix
            FFR: free-flow reachability matrix
            feature_size: the dimension of features
            Clamp_A: Boolean value, clamping all elements of A between 0. to 1.
        '''
        super(GraphConvolutionalLSTM, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = feature_size
        self.K = K
        
        self.A_list = [] # Adjacency Matrix List
        A = torch.FloatTensor(A)
        A_temp = torch.eye(feature_size,feature_size)
        for i in range(K):
            A_temp = torch.matmul(A_temp, torch.Tensor(A))
            if Clamp_A:
                # confine elements of A 
                A_temp = torch.clamp(A_temp, max = 1.) 
            #self.A_list.append(torch.mul(A_temp, torch.Tensor(FFR)))
            self.A_list.append(A_temp)
        
        # a length adjustable Module List for hosting all graph convolutions
        self.gc_list = nn.ModuleList([FilterLinear(feature_size, feature_size, self.A_list[i], bias=False) for i in range(K)])                  
        
        hidden_size = self.feature_size
        input_size = self.feature_size * K

        self.fl = nn.Linear(input_size + hidden_size, hidden_size)
        self.il = nn.Linear(input_size + hidden_size, hidden_size)
        self.ol = nn.Linear(input_size + hidden_size, hidden_size)
        self.Cl = nn.Linear(input_size + hidden_size, hidden_size)
        
        # initialize the neighbor weight for the cell state
        self.Neighbor_weight = Parameter(torch.FloatTensor(feature_size))
        stdv = 1. / math.sqrt(feature_size)
        self.Neighbor_weight.data.uniform_(-stdv, stdv)
        
    def forward(self, input, Hidden_State, Cell_State):
        x = input

        gc = self.gc_list[0](x)
        for i in range(1, self.K):
            gc = torch.cat((gc, self.gc_list[i](x)), 1)
            
        combined = torch.cat((gc, Hidden_State), 1)
        f = F.sigmoid(self.fl(combined))
        i = F.sigmoid(self.il(combined))
        o = F.sigmoid(self.ol(combined))
        C = F.tanh(self.Cl(combined))

        NC = torch.mul(Cell_State,  torch.mv(Variable(self.A_list[-1], requires_grad=False).cuda(), self.Neighbor_weight))
        Cell_State = f * NC + i * C
        Hidden_State = o * F.tanh(Cell_State)
        
        return Hidden_State, Cell_State, gc
    
    def Bi_torch(self, a):
        print("in GraphConvolutionalLSTM-Bi_torch")
        a[a < 0] = 0
        a[a > 0] = 1
        return a
    
    def loop(self, inputs):
        batch_size = inputs.size(0)
        time_step = inputs.size(1)
        Hidden_State, Cell_State = self.initHidden(batch_size)
        for i in range(time_step):
            Hidden_State, Cell_State, gc = self.forward(torch.squeeze(inputs[:,i:i+1,:]), Hidden_State, Cell_State)  
        return Hidden_State, Cell_State
    
    def initHidden(self, batch_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            return Hidden_State, Cell_State
        else:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size))
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_size))
            return Hidden_State, Cell_State
    def reinitHidden(self, batch_size, Hidden_State_data, Cell_State_data):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(Hidden_State_data.cuda(), requires_grad=True)
            Cell_State = Variable(Cell_State_data.cuda(), requires_grad=True)
            return Hidden_State, Cell_State
        else:
            Hidden_State = Variable(Hidden_State_data, requires_grad=True)
            Cell_State = Variable(Cell_State_data, requires_grad=True)
            return Hidden_State, Cell_State
            
#Actual train scenarios
def TrainGraphConvolutionalLSTM(train_dataloader, valid_dataloader, A, K, back_length = 3, num_epochs = 3, Clamp_A=False):
    
    inputs, labels = next(iter(train_dataloader))
    [batch_size, step_size, fea_size] = inputs.size()
    input_dim = fea_size
    hidden_dim = fea_size
    output_dim = fea_size
    print("in TRAINGraphConvolutionalLSTM")
    gclstm = GraphConvolutionalLSTM(K, torch.Tensor(A), A.shape[0], Clamp_A=Clamp_A)
    
    gclstm.cuda()
    
    loss_MSE = torch.nn.MSELoss()
    loss_L1 = torch.nn.L1Loss()
    learning_rate = 1e-5
    optimizer = torch.optim.RMSprop(gclstm.parameters(), lr = learning_rate)
    
    use_gpu = torch.cuda.is_available()
    
    interval = 50
    losses_train = []
    losses_interval_train = []
    losses_valid = []
    losses_interval_valid = []
    
    cur_time = time.time()
    pre_time = time.time()
    
    for epoch in range(num_epochs):
        #print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        #print('-' * 10)
        
        trained_number = 0
        
        # validation data loader iterator init
        valid_dataloader_iter = iter(valid_dataloader)
        
        for data in train_dataloader:
            inputs, labels = data

            if inputs.shape[0] != batch_size:
                continue

            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else: 
                inputs, labels = Variable(inputs), Variable(labels)
                
            
            gclstm.zero_grad()

            Hidden_State, Cell_State = gclstm.loop(inputs)
            
            loss_1 = loss_MSE(Hidden_State, labels)
            loss_2 = loss_L1(Hidden_State, labels)
            #criterion = calcSMAPE(Hidden_State, labels)
            loss_train = loss_1 + loss_2 #+ criterion
            optimizer.zero_grad()
            
            loss_train.backward()
            
            optimizer.step()
            
            losses_train.append(loss_train.data)
            
            # validation 
            try: 
                inputs_val, labels_val = next(valid_dataloader_iter)
            except StopIteration:
                valid_dataloader_iter = iter(valid_dataloader)
                inputs_val, labels_val = next(valid_dataloader_iter)
            
            if use_gpu:
                inputs_val, labels_val = Variable(inputs_val.cuda()), Variable(labels_val.cuda())
            else: 
                inputs_val, labels_val = Variable(inputs_val), Variable(labels_val)

            Hidden_State, Cell_State = gclstm.loop(inputs_val)
            loss_valid = loss_MSE(Hidden_State, labels)
            losses_valid.append(loss_valid.data)
            
            # output
            trained_number += 1
            
            if trained_number % interval == 0:
                cur_time = time.time()
                loss_interval_train = np.around(sum(losses_train[-interval:]).cpu().numpy()[0]/interval, decimals=8)
                losses_interval_train.append(loss_interval_train)
                loss_interval_valid = np.around(sum(losses_valid[-interval:]).cpu().numpy()[0]/interval, decimals=8)
                losses_interval_valid.append(loss_interval_valid)
                pre_time = cur_time

    return gclstm, [losses_train, losses_interval_train, losses_valid, losses_interval_valid]

taxi_demand_matrix =  pd.read_pickle('Demand/graphconvlstm/Graph_Convolutional_LSTM-master/Data/sg60.obj')
A = np.load('Demand/graphconvlstm/Graph_Convolutional_LSTM-master/Data/sp_g.npy')
A = A+(np.eye(np.shape(A)[0], dtype = int))

train_data, test_data, train_dataloader, valid_dataloader, test_dataloader, max_taxi_demand = PrepareDataset(taxi_demand_matrix)

losses_mase = []
losses_smape = []
losses_rmse = []

for rerunloop in range(5):
    gclstm, gclstm_loss = TrainGraphConvolutionalLSTM(train_dataloader, valid_dataloader, A, K=1, back_length = 2, num_epochs = 500, Clamp_A = True)
    #Actual testing scenario
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
        
        fromfull_mase_errors = []
        fromfull_smape_errors = []
        fromfull_rmse_errors = []
        Hidden_state = max_taxi_demand *Hidden_State.data.cpu() 
        np.savetxt("Demand/graphconvlstm/Graph_Convolutional_LSTM-master/Results/forHEDGE/REDO_sg60_yhat_l10_b24_st_"+str(rerunloop)+".csv",Hidden_state, fmt='%10.3f',delimiter = ",")
        Labels = max_taxi_demand*labels.data.cpu()
        np.savetxt("Demand/graphconvlstm/Graph_Convolutional_LSTM-master/Results/forHEDGE/REDO_sg60_y_l10_b24_st_"+str(rerunloop)+".csv",Labels[:,0,:], fmt='%10.3f',delimiter = ",")
        Train_data = max_taxi_demand*train_data
        for counter in range(np.shape(Hidden_State)[1]):
            mase = MASE(train_data[:,9,counter],Labels[:24,0,counter],Hidden_state[:24,counter])
            smape = SMAPE(Labels[:24,0,counter],Hidden_state[:24,counter])
            rmse = RMSE(Labels[:24,0,counter],Hidden_state[:24,counter])
            fromfull_mase_errors.append(mase)
            fromfull_smape_errors.append(smape)
            fromfull_rmse_errors.append(rmse)
            
        
        matches = read_csv("Demand/graphconvlstm/Graph_Convolutional_LSTM-master/Data/match_indx_740.csv", header = 0)
        matches.drop(matches.columns[[0]],axis =1 ,inplace = True)
        matches = matches.values
        errors = []
        for count in range(np.shape(matches)[0]): errors.append(fromfull_mase_errors[matches[count,0]])
        mase_perloader.append(errors)

        errors = []
        for count in range(np.shape(matches)[0]): errors.append(fromfull_smape_errors[matches[count,0]])
        smape_perloader.append(errors)

        errors = []
        for count in range(np.shape(matches)[0]): errors.append(fromfull_rmse_errors[matches[count,0]])
        rmse_perloader.append(errors)

        
        
    losses_mase.append(mase_perloader)
    losses_smape.append(smape_perloader)
    losses_rmse.append(rmse_perloader)


losses_mase = np.array(losses_mase)/max_taxi_demand
losses_smape = np.array(losses_smape)
losses_rmse = np.array(losses_rmse)

print('Tested: MASE_mean: {}, MASE_std : {}'.format(np.mean(losses_mase), np.std(losses_mase)))
print('Tested: SMAPE_mean: {}, SMAPE_std : {}'.format(np.mean(losses_smape), np.std(losses_smape)))
print('Tested: RMSE_mean: {}, RMSE_std : {}'.format(np.mean(losses_rmse), np.std(losses_rmse)))
    
    
np.savetxt("Demand/graphconvlstm/Graph_Convolutional_LSTM-master/Results/forHEDGE/REDO_sg60_mase_wsum2_l10_b24_st_rms.csv", losses_mase[:,0,:], fmt='%10.3f',delimiter = ",")
np.savetxt("Demand/graphconvlstm/Graph_Convolutional_LSTM-master/Results/forHEDGE/REDO_sg60_smape_wsum2_l10_b24_st_rms.csv", losses_smape[:,0,:], fmt='%10.3f',delimiter = ",")
np.savetxt("Demand/graphconvlstm/Graph_Convolutional_LSTM-master/Results/forHEDGE/REDO_sg60_rmse_wsum2_l10_b24_st_rms.csv", losses_rmse[:,0,:], fmt='%10.3f',delimiter = ",")
