dataset = read_csv("/media/data/users/kachD/Supply_dem Journal Revamp/fresh_start/HyperOpting/TRUE_SV60_740/S_TRUE_V60_fixedneigh8COR_740.csv", header=0)
dim3 = 9
dim1 = 740
dataset.drop(dataset.columns[[0]],axis =1 ,inplace = True)
input2LSTM = np.zeros((dim1,np.shape(dataset)[0]-1,dim3+1))
loop = 0
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
for sicounter in range(8):
    r = range(sicounter+1)
    r.append(-1)
    input2LSTM1 = input2LSTM[:,:,r]
    print(np.shape(input2LSTM1))
    dim3 = sicounter+1
    #if sicounter == 1:
     #   input2LSTM = input2LSTM[:,:,[0,-1]]
     #   dim3 = 1
    input2LSTMt = input2LSTM1.transpose([1,0,2])
    n_train_hours = np.shape(input2LSTMt)[0]-24
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
    print('batch size:',b_s)
    print('L1:', L1[hcounter])
    print('L2:', L2[hcounter])
    print('sicounter:', sicounter)
    maselist = []
    smapelist = []
    rmselist = []
    for rerunloop in range(5,10):
        print('rerunloop:', rerunloop)
        model = Sequential()
        model.add(LSTM(L1[hcounter],return_sequences=True,input_shape=(None,dim3)))
        model.add(Dropout(0.3685849187307607))
        if (L2[hcounter]!=0):
            model.add(LSTM(L2[hcounter],return_sequences=True))
            model.add(Dropout(0.3258984077443891))

        model.add(Dense(1,activation = 'linear'))#,return_sequences=True)) #-------------------ADD MORE DENSE LAYERS?
        adam = Adam(lr= 10**-1) 
        model.compile(loss='mse', optimizer=adam)
        history = model.fit(train_X_3d,train_y_3d,batch_size =b_s, validation_split = 0.1,epochs=training_epochs,callbacks = [stop],verbose=0,shuffle = False) #-------SAVE MODEL
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        pyplot.legend()
        pyplot.show()

        X_3d = np.zeros((np.shape(input2LSTM1)[0],np.shape(input2LSTM1)[1],np.shape(input2LSTM1)[2]-1))
        X_3d[:,:n_train_hours,:] = train_X_3d
        X_3d[:,n_train_hours:,:] = test_X_3d
        selfpredictions = model.predict(X_3d)
        if (np.isnan(selfpredictions).any() == True):
            print("getout")
            continue
        selfpredictions = selfpredictions[:,-np.shape(test_y_3d)[1]:,:]  
        to_transpose = np.concatenate((test_X_3d,selfpredictions), axis = 2)
        transposed = to_transpose.transpose([1,0,2])
        to_invert = transposed.reshape(np.shape(transposed)[0], np.shape(transposed)[1]*np.shape(transposed)[2])
        inverted = scaler_train.inverse_transform(to_invert)
        predictions_3d = inverted.reshape(np.shape(test)[0], np.shape(test)[1], np.shape(test)[2])
        predictions_3d = predictions_3d.transpose([1,0,2])
        fromfullpred_actuals = predictions_3d[:,:,-1]
        model.save("/media/data/users/kachD/Supply_dem Journal Revamp/fresh_start/HyperOpting/TRUE_SV60_740/S_TRUE_V60_fixedneigh8COR_740"+"_"+str(sicounter)+"_"+str(rerunloop)+".h5")
        np.savetxt("/media/data/users/kachD/Supply_dem Journal Revamp/fresh_start/HyperOpting/TRUE_SV60_740/S_TRUE_V60_fixedneigh8COR_740_pred_"+str(sicounter)+"_"+str(rerunloop)+".csv", np.array(fromfullpred_actuals), fmt='%10.3f',delimiter = ",")


        test_y = input2LSTM1[:,n_train_hours:,-1]
        fromfull_mase_errors = []
        fromfull_smape_errors = []
        fromfull_rmse_errors = []
        for counter in range(np.shape(test_X_3d)[0]):
            mase = MASE(input2LSTM1[counter,:n_train_hours,0],test_y[counter,:],fromfullpred_actuals[counter,:])
            smape = SMAPE(test_y[counter,:],fromfullpred_actuals[counter,:])
            rmse = RMSE(test_y[counter,:],fromfullpred_actuals[counter,:])
            fromfull_mase_errors.append(mase)
            fromfull_smape_errors.append(smape)
            fromfull_rmse_errors.append(rmse)
        
        np.savetxt("/media/data/users/kachD/Supply_dem Journal Revamp/fresh_start/HyperOpting/TRUE_SV60_740/S_TRUE_V60_fixedneigh8COR_740_mase_"+str(sicounter)+"_"+str(rerunloop)+".csv", np.array(fromfull_mase_errors), fmt='%10.3f',delimiter = ",")
        np.savetxt("/media/data/users/kachD/Supply_dem Journal Revamp/fresh_start/HyperOpting/TRUE_SV60_740/S_TRUE_V60_fixedneigh8COR_740_smape_"+str(sicounter)+"_"+str(rerunloop)+".csv", np.array(fromfull_smape_errors), fmt='%10.3f',delimiter = ",")
        np.savetxt("/media/data/users/kachD/Supply_dem Journal Revamp/fresh_start/HyperOpting/TRUE_SV60_740/S_TRUE_V60_fixedneigh8COR_740_rmse_"+str(sicounter)+"_"+str(rerunloop)+".csv", np.array(fromfull_rmse_errors), fmt='%10.3f',delimiter = ",")

        maselist.append(fromfull_mase_errors)
        smapelist.append(fromfull_smape_errors)
        rmselist.append(fromfull_rmse_errors)

        print('from full mase errors', np.asarray(fromfull_mase_errors).mean())
        print('from full smape errors', np.asarray(fromfull_smape_errors).mean())
        print('from full rmse errors', np.asarray(fromfull_rmse_errors).mean())
        print('from full mase errors 1:50', np.asarray(fromfull_mase_errors)[1:50].mean())
        print('from full smape errors 1:50', np.asarray(fromfull_smape_errors)[1:50].mean())
        print('from full rmse errors 1:50', np.asarray(fromfull_rmse_errors)[1:50].mean())
        print('from full mase errors -50:', np.asarray(fromfull_mase_errors)[-50:].mean())
        print('from full smape errors -50:', np.asarray(fromfull_smape_errors)[-50:].mean())
        print('from full rmse errors -50:', np.asarray(fromfull_rmse_errors)[-50:].mean())
    maselist_array = np.array(maselist)
    smapelist_array = np.array(smapelist)
    rmselist_array = np.array(rmselist)
    np.savetxt("/media/data/users/kachD/Supply_dem Journal Revamp/fresh_start/HyperOpting/TRUE_SV60_740/S_TRUE_V60_fixedneigh8COR_740_5runs_mase_"+str(sicounter)+".csv", maselist_array, fmt='%10.3f',delimiter = ",")
    np.savetxt("/media/data/users/kachD/Supply_dem Journal Revamp/fresh_start/HyperOpting/TRUE_SV60_740/S_TRUE_V60_fixedneigh8COR_740_5runs_smape_"+str(sicounter)+".csv", smapelist_array, fmt='%10.3f',delimiter = ",")
    np.savetxt("/media/data/users/kachD/Supply_dem Journal Revamp/fresh_start/HyperOpting/TRUE_SV60_740/S_TRUE_V60_fixedneigh8COR_740_5runs_rmse_"+str(sicounter)+".csv", rmselist_array, fmt='%10.3f',delimiter = ",")

