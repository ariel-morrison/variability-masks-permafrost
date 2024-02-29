#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 12:42:01 2022

@author: Ariel L. Morrison

----------------------------------
Function: train logistic regression neural net
    to predict which simulation a map of active
    layer depth is from (control or ARISE)
"""

def trainModel(timeFrame,var,numYearsTrain):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras import regularizers
    from tensorflow.keras import metrics
    from tensorflow.keras import optimizers
    from tensorflow.keras.models import Sequential
    from preprocessingData import preprocessDataForPrediction
    datadir = '/Users/arielmor/Desktop/SAI/data/ARISE/data/model_output'
    
    
    ## ------ random numbers ------ ##
    np.random.seed(99)
    tf.random.set_seed(99)
    
    ## ------ Choose parameters ------ ##
    timeFrame   = str(timeFrame)
    var = str(var)
    batch_size  = 28
    verbose     = 2
    singleLayer = True
    denseShape  = 28
    dropout     = 0.25
    if var == 'ALT':
        epochs      = 3 
        lr          = 0.1 
        l1, l2      = 0.002,0.9 
    elif var == 'ALTMAX':
        epochs = 3
        lr     = 0.01 
        l1, l2 = 0.01,0.94 
        
    elif var == 'NEE' or var == 'GPP':
        epochs  = 5
        lr      = 0.08
        l1, l2  = 0.015,1
    elif var == 'TG':
        batch_size  = 28
        epochs      = 6 
        lr          = 0.1 
        l1, l2      = 0.01,0.7 
    elif var == 'ER':
        epochs  = 5
        lr      = 0.08
        l1, l2  = 0,0.9
    elif var == 'TSOI3m':
        if timeFrame == 'summer':
            epochs  = 3 
            lr      = 0.009 
            l1, l2  = 0.0045,0.9 
        else:
            epochs = 4
            lr     = 0.12
            l1, l2 = 0.001, 0.95
    elif var == 'TSOI':
        epochs = 3
        lr     = 0.08 
        l1, l2 = 0.001, 0.75 
    elif var == 'FSNO':
        epochs  = 6
        lr      = 0.1
        l1, l2  = 0.01, 0.9
    else:
        epochs  = 3 
        lr      = 0.001 
        l1, l2  = 0.04,0.6 
    ## -------------------------------- ## 
    
    
    ## ------ Get processed training and test data ------ ##
    lat,lon,features_train,features_val,features_test,\
        labels_train,labels_val,labels_test,\
        X_train,y_train,\
        X_val,y_val,\
        X_test,y_test,\
        lenTime,testMemNum = preprocessDataForPrediction(str(
                                                    timeFrame),
                                                    str(var),numYearsTrain)
    ## ---------------------------------------------------- ## 
    
    
    ## ------ Create and train the model ------ ##
    tf.keras.backend.clear_session()
    rseed = 4444
    tf.keras.utils.set_random_seed(rseed)
    model = Sequential()
    
    if singleLayer:
        model.add(Dense(y_train.shape[1], input_shape=(X_train.shape[1],),
                            activation         = 'sigmoid',
                            use_bias           = True,
                            kernel_regularizer = regularizers.l1_l2(l1=l1, l2=l2),
                            bias_initializer   = tf.keras.initializers.LecunNormal(seed=rseed),
                            kernel_initializer = tf.keras.initializers.LecunNormal(seed=rseed)))
        
    else:
        model.add(Dense(denseShape, input_shape=(X_train.shape[1],),
                        activation         = 'relu',
                        use_bias           = True,
                        kernel_regularizer = regularizers.l1_l2(l1=l1, l2=l2),
                        bias_initializer   = tf.keras.initializers.LecunNormal(seed=rseed),
                        kernel_initializer = tf.keras.initializers.LecunNormal(seed=rseed)))
        model.add(Dropout(dropout))
        model.add(Dense(y_train.shape[1],
                        activation         = 'softmax',
                        use_bias           = True,
                        kernel_regularizer = regularizers.l1_l2(l1=l1, l2=l2),
                        bias_initializer   = tf.keras.initializers.LecunNormal(seed=rseed),
                        kernel_initializer = tf.keras.initializers.LecunNormal(seed=rseed)))
    
    ''' Schedule a decreasing learning rate '''
    def scheduler(epoch, lr):
        if epoch < 1 or epoch > 15:
            return lr
        else:
            if var == 'TSOI_int3m':
                return lr*0.1
            elif var == 'ALTMAX':
                return lr/3.33
            else: return lr/3.
    ## ------------------------------- ## 
    
        
    ## ------ Compile the model ------ ##   
    model.compile(optimizer = optimizers.SGD(learning_rate = lr, 
                                              momentum     = 0.71, 
                                              nesterov     = True),
                  loss      = tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics   = [tf.keras.metrics.binary_accuracy,]) # binary_accuracy,
    
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler) 
    stopEarly = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)   
    print(model.summary()); print(" ")
    
    history = model.fit(X_train, 
                        y_train, 
                        batch_size      = batch_size, 
                        epochs          = epochs, 
                        shuffle         = True, 
                        verbose         = verbose, 
                        callbacks       = [lr_scheduler,stopEarly,],
                        validation_data = (X_val,y_val))
    ## -------------------------------- ## 
    
    
    ## ------ Predictions! ------ ##
    y_pred_train = model.predict(X_train)
    y_pred_val   = model.predict(X_val)
    y_pred_test  = model.predict(X_test)
    ## -------------------------- ## 
    
    
    ## ------ Prediction confidence ------ ##
    '''
        y_pred_test shape is two columns
        1st column = probability that map comes from CONTROL
        2nd column = probability that map comes from FEEDBACK
        Length of array = 2 x lenTime
        20 years of training data = 40 predictions --> 1st 20 = control, 2nd 20 = feedback
    '''
    y_pred_CONTROL= np.round(y_pred_test[:lenTime],decimals=3)
    y_pred_FEEDBACK = np.round(y_pred_test[lenTime:],decimals=3)
    np.save(datadir + '/y_pred_CONTROL_' + str(testMemNum) + '_' + str(var) + '_' + str(timeFrame) + '.npy',
            y_pred_CONTROL)
    np.save(datadir + '/y_pred_FEEDBACK_' + str(testMemNum) + '_' + str(var) + '_' + str(timeFrame) + '.npy',
            y_pred_FEEDBACK)
    
    ## add markers to signify if prediction is right or wrong ##
    df = pd.DataFrame.from_dict(dict(controlprediction=(1-y_pred_CONTROL[:,0]),
                                     feedbackprediction=y_pred_FEEDBACK[:,0]))
    df.index.name="index"
    df.reset_index(inplace=True)
    df["criteria"] = df.controlprediction > 0.5
    df["criteria2"] = df.feedbackprediction > 0.5
    ## make figure ##
    fig, ax = plt.subplots(figsize=(8,4),dpi=1200)
    ax = df.controlprediction.plot(c='r',label='SSP2-4.5',linewidth=1)
    df.feedbackprediction.plot(c='xkcd:cobalt blue',ax=ax,label='ARISE',linewidth=1)
    df[df.criteria].plot(kind='scatter',x='index',y='controlprediction',
                                 ax=ax,c='r',marker='o')
    df[~df.criteria].plot(kind='scatter',x='index',y='controlprediction',
                                  ax=ax,c='r',marker='x')
    df[df.criteria2].plot(kind='scatter',x='index',y='feedbackprediction',
                                  ax=ax,c='xkcd:cobalt blue',marker='o')
    df[~df.criteria2].plot(kind='scatter',x='index',y='feedbackprediction',
                                   ax=ax,c='xkcd:cobalt blue',marker='x') 
    if timeFrame == 'winter' or timeFrame == 'non_growing_season':
        plt.xlim([0,lenTime-1])
        ax.set_xticks([0,5,10,15,20,25,30,34])
        ax.set_xticklabels(['2035','2040','2045','2050','2055','2060','2065','2069'])
    elif timeFrame == 'monthly':
        plt.xlim([0,lenTime])
    else:
        plt.xlim([0,lenTime-1])
        ax.set_xticks([0,5,10,15,20,25,30,34])
        ax.set_xticklabels(['2035','2040','2045','2050','2055','2060','2065','2069'])
    plt.axhline(0.5,0,lenTime,linestyle='dashed',color='k',linewidth=0.6)
    plt.legend(loc='lower right')
    ax.set_yticks([])
    ax.set(xlabel=None)
    plt.ylabel('Prediction confidence\n incorrect                        correct',
               fontweight='bold')
    plt.ylim(0,1)
    ax.fill_between(range(lenTime),0,0.5,alpha=0.1,color='k')
    plt.savefig('/Users/arielmor/Desktop/SAI/data/ARISE/figures/prediction_confidence_'\
                +str(timeFrame)+'_'+str(var)+'.jpg',bbox_inches='tight',dpi=1200)
    plt.show()
    ## ------------------------------------------------------------------------- ##
    
    ## ------ Accuracy metrics ------ ##
    print("binary accuracy: ", np.round(history.history['binary_accuracy'],decimals=3))
    
    ''' Accuracy of each prediction vs label '''
    acc_CONTROL = np.asarray(np.asarray(np.equal(np.round(np.squeeze(y_pred_CONTROL)),
                                                 labels_test[0,:lenTime]),dtype='int32'),dtype='float')
    acc_FEEDBACK = np.asarray(np.asarray(np.equal(np.round(np.squeeze(y_pred_FEEDBACK)),
                                                  labels_test[1,:lenTime]),dtype='int32'),dtype='float')
    
    print("-------------------------------------------")
    print("Correct predictions (label = 1; above 50% confidence)")
    print("CONTROL", acc_CONTROL)
    print("FEEDBACK", acc_FEEDBACK)
    print("-------------------------------------------")
    
    fig, (ax1, ax2) = plt.subplots(2, figsize=(8,4), dpi=900)
    ax2.scatter(range(lenTime),acc_FEEDBACK,color='b',label='ARISE-SAI')
    ax2.legend(loc="lower right", fancybox=True, fontsize=11)
    ax1.scatter(range(lenTime),acc_CONTROL,color='r',label='SSP2-4.5')
    ax1.legend(loc="lower right", fancybox=True, fontsize=11)
    if timeFrame == 'winter' or timeFrame == 'non_growing_season':
        ax1.set_xticks([0,5,10,15,20,25,30,34])
        ax1.set_xticklabels([])
        ax2.set_xticks([0,5,10,15,20,25,30,34])
        ax2.set_xticklabels(['2035','2040','2045','2050','2055','2060','2065','2069'])
    else:
        ax1.set_xticks([0,5,10,15,20,25,30,34])
        ax1.set_xticklabels([])
        ax2.set_xticks([0,5,10,15,20,25,30,34])
        ax2.set_xticklabels(['2035','2040','2045','2050','2055','2060','2065','2069'])
    ax2.set_yticks([0,1]) 
    ax2.set_yticklabels(['False','True'], fontweight='bold')
    ax1.set_yticks([0,1]) 
    ax1.set_yticklabels(['False','True'], fontweight='bold')
    plt.savefig('/Users/arielmor/Desktop/SAI/data/ARISE/figures/prediction_accuracy_'\
                +str(timeFrame)+'_'+str(var)+'.jpg', bbox_inches='tight',dpi=900); plt.show()
    ## ----------------------------------------------------------------------- ## 
    
    ## ------ Summarize history for loss ------ ##
    plt.figure(figsize=(8,4),dpi=700)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss ' + '(' + str(timeFrame) + ') ' + str(var), fontsize=11)
    plt.ylabel('loss'); plt.xlabel('epoch')
    plt.xlim([0,epochs-1])
    plt.legend(['train', 'val'], loc='upper right')
    plt.savefig('/Users/arielmor/Desktop/SAI/data/ARISE/figures/train_val_loss_'\
                +str(timeFrame)+'_'+str(var)+'.jpg',bbox_inches='tight',dpi=700); plt.show()
    ## ---------------------------------------- ---------------- ## 
    '''
    First number is likelihood of map coming from first column of testing data,
    second number is likelihood of second column 
    
    First 20 numbers = control, second 20 = arise
    '''
    ## ------ Map of weights ------ ##
    if singleLayer:
        from plottingFunctions import get_colormap, make_maps
        lenLat = 37; lenLon = 288; 
        brbg_cmap,rdbu_cmap,jet,magma,reds,hot,seismic = get_colormap(21)
        mapWeights = (model.layers[0].get_weights()[0].reshape(lenLat,lenLon))
        np.save('map_weights_' + str(timeFrame) + '_' + str(var) + '.npy', mapWeights)
        vmins = -0.03; vmaxs = 0.03
        fig,ax = make_maps(mapWeights,lat[:-7],lon,
                            vmins,vmaxs,21,rdbu_cmap,
                            'weights','weights for '+str(timeFrame) + ', ' + str(l1) + ', ' + str(l2),
                            'weights_'+str(timeFrame)+'_'+str(var),'both',False,False)
        # mapWeights = model.layers[0].get_weights()[0][:,1].reshape(lenLat,lenLon)
        # fig,ax = make_maps(mapWeights,lat[29:-6],lon,
        #                     -0.02,0.02,21,brbg_cmap,'weights','weights for '+str(timeFrame)+', control','feedback_weights_'+str(timeFrame))
    ## ------------------------------------------------------------ ##
    
    model.save('pfrost_down_to_bedrock_logistic_regression_predict_'+str(var)+'_'+str(timeFrame))
    # del model
    return model



