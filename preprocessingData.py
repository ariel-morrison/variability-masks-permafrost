#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ariel L. Morrison
""" 
def preprocessDataForPrediction(timeFrame,var,numYears):
    import os; os.chdir('/Users/arielmor/Projects/actm-sai-csu/research/arise_arctic_climate')
    from tippingPoints import findPermafrost
    import numpy as np
    import warnings
    
    datadir = '/Users/arielmor/Desktop/SAI/data/ARISE/data'
    
    #### READ ANNUAL MEAN FIELDS
    if var == 'ALT':
        controlANN,control,altmaxAnnCONTROL,lat,lon = findPermafrost(
            'b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.', '201501-206912')
        ariseANN,arise,altmaxAnnFEEDBACK,lat,lon = findPermafrost(
            'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.', '203501-206912')
    elif var == 'TSOI':
        import xarray as xr
        ds = xr.open_dataset(datadir + 
                        '/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001.clm2.h0.ALT.203501-206912_NH.nc')
        lat = ds.lat[41:]
        lon = ds.lon
        ds.close()
        import pickle
        with open('/Users/arielmor/Desktop/SAI/data/ARISE/data/permafrost_temperature_FEEDBACK.pkl', 'rb') as fp:
            ariseANN = pickle.load(fp)
        with open('/Users/arielmor/Desktop/SAI/data/ARISE/data/permafrost_temperature_CONTROL.pkl', 'rb') as fp:
            controlANN   = pickle.load(fp)
        
    
    #### PROCESS DATA
    ''' 
    lenTime = number of time units in 35 years
    numYears = number of years for training (training on 35 years)
    numUnits = total number of time units (e.g., total years in data record)
    '''
    timeStartControl = 20
    varCONTROL   = controlANN
    varFEEDBACK  = ariseANN
    
    lenTime          = 1*numYears
    numUnits         = 35 # number of total years
    
   
    ## ---------------------------------------------------- ##
    new_list = ['002', '010', '003', '004', '008', '007', '009', 
                '005', '006', 
                '001']
    
    #### MAKE FEATURES
    # CONTROL
    featuresC = []
    for ensNum in range(len(new_list)):
        if timeFrame == 'winter' or timeFrame == 'non_growing_season':
            temp = np.stack(varCONTROL[new_list[ensNum]][timeStartControl:,:-7,:].astype('float'))
        else:
            if var == 'ALT' or var == 'ALTMAX' or var == 'TSOI3m' or var == 'TG' or var == 'SNOW_DEPTH' or var == 'FSNO':
                temp = np.stack(varCONTROL[new_list[ensNum]][20:,:-7,:].values.astype('float'))
            elif var == 'TSOI':
                temp = np.stack(varCONTROL[new_list[ensNum]][20:,:-7,:].astype('float'))
        featuresC.append(temp)
    del temp
    
    # FEEDBACK
    featuresF = []
    for ensNum in range(len(new_list)):
        if timeFrame == 'winter' or timeFrame == 'non_growing_season':
            temp = np.stack(varFEEDBACK[new_list[ensNum]][:,:-7,:].astype('float'))
        else:
            if var == 'ALT' or var == 'ALTMAX' or var == 'TSOI3m' or var == 'TG' or var == 'SNOW_DEPTH' or var == 'FSNO':
                temp = np.stack(varFEEDBACK[new_list[ensNum]][:,:-7,:].values.astype('float'))
            elif var == 'TSOI':
                temp = np.stack(varFEEDBACK[new_list[ensNum]][:,:-7,:].astype('float'))
        featuresF.append(temp)
    del temp
    ## ---------------------------------------------------- ##
    
    #### SPLIT TRAINING/TEST DATA
    valNum  = 2
    testNum = 1
    print('training members = ' + str(new_list[:-(valNum+testNum)]))
    print('validate members = ' + str(new_list[-(valNum+testNum):-testNum]))
    print('testing member = ' + str(new_list[-testNum:]))
    
    testMemNum      = np.asarray(new_list[-testNum:]); 
    testMemNum      = testMemNum.astype(float)
    
    featuresF_train = featuresF[:-(valNum+testNum)]
    featuresC_train = featuresC[:-(valNum+testNum)]
    featuresF_val   = featuresF[-(valNum+testNum):-testNum]
    featuresC_val   = featuresC[-(valNum+testNum):-testNum]
    featuresF_test  = featuresF[-testNum:]
    featuresC_test  = featuresC[-testNum:]
    del new_list
    ## ---------------------------------------------------- ##
    
    #### CREATE LABELS
    ''' 
        number of ensemble members, number of time steps
        CONTROL = 0; FEEDBACK = 1
    '''
    labelsC_train = np.ones((len(featuresC_train),numUnits))*0
    labelsF_train = np.ones((len(featuresF_train),numUnits))*1
    labelsC_val   = np.ones((len(featuresC_val),numUnits))*0
    labelsF_val   = np.ones((len(featuresF_val),numUnits))*1
    labelsC_test  = np.ones((len(featuresC_test),numUnits))*0
    labelsF_test  = np.ones((len(featuresF_test),numUnits))*1
    ## ---------------------------------------------------- ##
    
    ## ------ Concatenate control/feedback data for neural net training ------ ##
    '''
        also important for standardizing data
    '''
    features_train = np.append(featuresC_train,featuresF_train,axis=0)
    features_val   = np.append(featuresC_val,featuresF_val,axis=0)
    features_test  = np.append(featuresC_test,featuresF_test,axis=0)
    
    labels_train   = np.append(labelsC_train,labelsF_train,axis=0)
    labels_val     = np.append(labelsC_val,labelsF_val,axis=0)
    labels_test    = np.append(labelsC_test,labelsF_test,axis=0)
    ## ----------------------------------------------------------------------- ##
    
    #### STANDARDIZE DATA
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        featuresMean = np.nanmean(features_train,axis=(0,1))
        featuresStd  = np.nanstd(features_train,axis=(0,1))
        
        ''' Standardize training, validation, testing the same way '''
        features_train = (features_train-featuresMean)/featuresStd
        features_val = (features_val-featuresMean)/featuresStd
        features_test = (features_test-featuresMean)/featuresStd       
    ## ------------------------------ ##
    
    ## ------ Replace NaNs with 0 ------ ##
    features_train[np.isnan(features_train)] = 0.
    features_val[np.isnan(features_val)]     = 0.
    features_test[np.isnan(features_test)]   = 0.
    ## --------------------------------- ##
    
    ## ------ Categorical labels ------ ##
    import tensorflow as tf
    y_train = tf.keras.utils.to_categorical(labels_train)
    y_val   = tf.keras.utils.to_categorical(labels_val)
    y_test  = tf.keras.utils.to_categorical(labels_test)
    ## ---------------------------------------------------- ##    
    
    ## ------ Select training years ------ ##
    # Training years = 2035-2069
    X_train = features_train[:,:lenTime,:,:]
    X_val   = features_val[:,:lenTime,:,:]
    X_test  = features_test[:,:lenTime,:,:]
    # y_train = y_train[:,:lenTime,0]
    # y_val   = y_val[:,:lenTime,0]
    # y_test  = y_test[:,:lenTime,0]
    y_train = y_train[:,:lenTime]
    y_val   = y_val[:,:lenTime]
    y_test  = y_test[:,:lenTime]
    ## ---------------------------------------------------- ## 
    
    ## ------ Flatten data ------ ##
    lenLat = 37; lenLon = 288; 
    
    X_train = X_train.reshape(len(X_train)*lenTime,lenLat*lenLon)
    X_val   = X_val.reshape(len(X_val)*lenTime,lenLat*lenLon)
    X_test  = X_test.reshape(len(X_test)*lenTime,lenLat*lenLon)
    
    y_train = y_train.reshape((len(y_train)*lenTime,2))
    y_val   = y_val.reshape((len(y_val)*lenTime,2))
    y_test  = y_test.reshape((len(y_test)*lenTime,2))
    ## ---------------------------------------------------- ## 
    
    ## ------ Replace NaNs and inf ------ ##  
    from numpy import inf
    X_train[np.isnan(X_train)] = 0.
    X_train[X_train == inf]    = 0.
    X_train[X_train == -inf]   = 0.
    X_val[np.isnan(X_val)]     = 0.
    X_val[X_val == inf]        = 0.
    X_val[X_val == -inf]       = 0.
    X_test[np.isnan(X_test)]   = 0.
    X_test[X_test == inf]      = 0.
    X_test[X_test == -inf]     = 0.
    y_train[np.isnan(y_train)] = 0.
    y_val[np.isnan(y_val)]     = 0.
    y_test[np.isnan(y_test)]   = 0.
    ## ---------------------------------------------------- ## 
    
    #### PREDICT IF DATA COME FROM CONTROL SIM
    y_train = y_train[:,1:]
    y_val   = y_val[:,1:]
    y_test  = y_test[:,1:]
    ## ---------------------------------------------------- ## 
    
    
    return lat,lon,features_train,features_val,features_test,\
        labels_train,labels_val,labels_test,X_train,y_train,X_val,\
            y_val,X_test,y_test,lenTime,testMemNum
