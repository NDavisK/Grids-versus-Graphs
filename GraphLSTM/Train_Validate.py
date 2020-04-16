
import torch
import numpy as np
from torch.autograd import Variable
import time
from Models import * 

def TrainGraphLSTM(train_dataloader, valid_dataloader, A, FFR, K, back_length = 3, num_epochs = 3, Clamp_A=False):
    
    inputs, labels = next(iter(train_dataloader))
    [batch_size, step_size, fea_size] = inputs.size()
    input_dim = fea_size
    hidden_dim = fea_size
    output_dim = fea_size
    print("in TRAINGraphLSTM")
    gclstm = GraphLSTM(K, torch.Tensor(A), FFR[back_length], A.shape[0], Clamp_A=Clamp_A)
    
    gclstm.cuda()
    
    loss_MSE = torch.nn.MSELoss()
    loss_L1 = torch.nn.L1Loss()
    
    learning_rate = 1e-5
    optimizer = torch.optim.RMSprop(gclstm.parameters(), lr = learning_rate)
    
    use_gpu = torch.cuda.is_available()
    
    interval = 100
    losses_train = []
    losses_interval_train = []
    losses_valid = []
    losses_interval_valid = []
    
    cur_time = time.time()
    pre_time = time.time()
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        trained_number = 0
        
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
            
            loss_train = loss_MSE(Hidden_State, labels)
            
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


def TestGraphLSTM(gclstm, test_dataloader, max_speed):
    inputs, labels = next(iter(test_dataloader))
    [batch_size, step_size, fea_size] = inputs.size()
    print("in TestGraphLSTM")

    cur_time = time.time()
    pre_time = time.time()
    
    use_gpu = torch.cuda.is_available()
    
    loss_MSE = torch.nn.MSELoss()
    loss_L1 = torch.nn.L1Loss()
    
    tested_batch = 0
    
    losses_mse = []
    losses_l1 = [] 
    
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
    
        tested_batch += 1
    
        if tested_batch % 1000 == 0:
            cur_time = time.time()
            print('Tested #: {}, loss_l1: {}, loss_mse: {}, time: {}'.format( \
                  tested_batch * batch_size, \
                  np.around([loss_l1.data[0]], decimals=8), \
                  np.around([loss_mse.data[0]], decimals=8), \
                  np.around([cur_time - pre_time], decimals=8) ) )
            pre_time = cur_time
    
    losses_l1 = np.array(losses_l1)
    
    losses_mse = np.array(losses_mse)
    
    mean_l1 = np.mean(losses_l1) * max_speed
    
    std_l1 = np.std(losses_l1) * max_speed
    
    print('Tested: L1_mean: {}, L1_std : {}'.format(mean_l1, std_l1))
    
    return [losses_l1, losses_mse, mean_l1, std_l1]
