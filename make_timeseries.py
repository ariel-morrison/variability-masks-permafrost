#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: arielmor
"""

def make_timeseries(numEns,var,lat,lon,latmax,latmin,lonmax,lonmin,dataDict):
    import numpy as np
    import warnings

    # restrict to only poleward of 50N        
    if len(lat) > 60: lat = lat[41:]
    
    lengthDictionary = len(dataDict)
    
    if lengthDictionary > 1: ens = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010']
    else: ens = ['001']
    numEns = len(ens)
    
    # for regional time series
    latmin_ind = int(np.abs(latmin-lat).argmin())
    latmax_ind = int(np.abs(latmax-lat).argmin())+1
    lonmin_ind = int(np.abs(lonmin-lon).argmin())
    lonmax_ind = int(np.abs(lonmax-lon).argmin())+1
    print(latmin_ind, latmax_ind)
    print(lonmin_ind, lonmax_ind)
    
    # Latitude weighting
    lonmesh,latmesh = np.meshgrid(lon,lat)
    
    # Mask out ocean and non-permafrost land:
    weights2D = {}
    for i in range(numEns):
        weights2D[ens[i]] = np.full((dataDict[ens[i]].shape[0],len(lat),len(lon)),np.nan) 
        for iyear in range(dataDict[ens[0]].shape[0]):
            weights2D[ens[i]][iyear,:,:] = np.cos(np.deg2rad(latmesh))
            weights2D[ens[i]][iyear,:,:][np.isnan(dataDict[ens[i]][iyear,:,:])] = np.nan
    
    
    # Annual time series for each ensemble member
    ensMemberTS = {}
    for ensNum in range(numEns):
        warnings.simplefilter("ignore")
                                            
        ensMasked         = dataDict[ens[ensNum]]
        ensMasked_grouped = ensMasked[:,latmin_ind:latmax_ind,lonmin_ind:lonmax_ind]
        ensMasked_grouped = np.ma.MaskedArray(ensMasked_grouped, mask=np.isnan(ensMasked_grouped))
        weights           = np.ma.asanyarray(weights2D[ens[ensNum]][
                                :,latmin_ind:latmax_ind,lonmin_ind:lonmax_ind])
        weights.mask      = ensMasked_grouped.mask
        ensMemberTS[ens[ensNum]] = np.array([np.ma.average(
                                            ensMasked_grouped[i],
                                            weights=weights[i]
                                            ) for i in range((ensMasked_grouped.shape)[0])])
    return ensMemberTS


def make_ensemble_mean_timeseries(ensMemberTS,numEns):
    ensMeanTS = 0
    if type(ensMemberTS).__module__ == 'numpy':
        for val in ensMemberTS:
            ensMeanTS += val
        ensMeanTS = ensMeanTS/numEns 
    else:
        for val in ensMemberTS.values():
            ensMeanTS += val
        ensMeanTS = ensMeanTS/numEns 
    return ensMeanTS
