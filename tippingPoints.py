#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 14:03:08 2022

@author: Ariel L. Morrison

"Tipping points" in the climate system are thresholds that, once
crossed, lead to sustained and irreversible warming. A tipping point
can also be a threshold beyond which the initial conditions of a system
cannot be recovered. Permafrost has a 'tipping point' threshold of ~0degC
(the thawing temperature) because it's hard to refreeze permafrost once
it has been thawed. Talik formation is a tipping point because it promotes
increased soil respiration and soil carbon loss (i.e., makes soil carbon
more accessible to decomposition and respiration.)
"""

def talikFormation(lat_lon_series, N):
    ## N = number of years that soil is thawed (warmer than -0.5C (Parazoo et al., 2018))
    ## for talik formation, needs to be perenially thawed
    ## so N = number of years left in the simulation
     import numpy as np
     mask = np.convolve(np.greater(lat_lon_series,272.65),np.ones(N,dtype=int))>=N
     if mask.any():
         return mask.argmax() - N + 1
     else:
         return None
     
        
def readDataForTippingPoints(var, middleAtm, controlSim, simulation, timePeriod, lnd, warmPfrost):
    import numpy as np
    import xarray as xr
    import pandas as pd
    
    datadir = '/Users/arielmor/Desktop/SAI/data/ARISE/data'

    myvar = {}
    myvarAnn = {}
    
    if middleAtm:
        if controlSim == True:
            ens = ['002']
        else:
            ens = ['002']
    else:
        ens = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010']
        
    numEns = len(ens)
    print("reading " + str(var) + " for " + str(simulation))
    
    if lnd:
        for i in range(numEns):
            ds = xr.open_dataset(datadir + '/' + str(simulation) + str(ens[i]) +
                                 '.clm2.h0.' + str(var) + '.' + 
                                 str(timePeriod) + '_NH.nc', decode_times=False)
            units, reference_date = ds.time.attrs['units'].split('since')
            ds['time'] = pd.date_range(start=reference_date, periods=ds.sizes['time'], freq='MS')
            ## restrict to poleward of 50N ##
            lat = ds.lat[41:]
            lon = ds.lon
            if var == 'TSOI3m':
                myvar[ens[i]] = np.squeeze(ds[str(var)][:,:,41:])
            ## get coldest monthly mean 3m soil temperature for each year ##
                myvarAnn[ens[i]] = myvar[ens[i]].groupby('time.year').min(dim='time', skipna=True) 
            elif var == 'TSOI':
                myvar[ens[i]] = ds[str(var)][:,:20,41:,:]
            ## get COLDEST monthly mean soil temp for each year ##
            ## this is for talik formation ##
            ## the coldest temp of the year needs to be > -0.5deg to be talik ##
                myvarAnn[ens[i]] = myvar[ens[i]].groupby('time.year').min(dim='time', skipna=True)
            
            elif var == 'TSOI_int3m':
                myvar[ens[i]] = ds[str(var)][:,41:,:]
                myvarAnn[ens[i]] = myvar[ens[i]].groupby('time.year').mean(dim='time', skipna=True)
            elif var == 'ALTMAX' or var == 'ALT':
                myvar[ens[i]] = ds[str(var)][:,41:,:]
                if warmPfrost:
                    myvar[ens[i]] = myvar[ens[i]].where(myvar[ens[i]] <= 9.)
                else:
                    myvar[ens[i]] = myvar[ens[i]].where(myvar[ens[i]] <= 3.5)
                    ## get maximum alt for each year ##
                if var == 'ALTMAX':
                    myvarAnn[ens[i]] = myvar[ens[i]].groupby('time.year').max(dim='time', skipna=True) 
                elif var == 'ALT':
                    myvarAnn[ens[i]] = myvar[ens[i]].groupby('time.year').mean(dim='time', skipna=True) 
            elif var == 'NBP' or var == 'ER' or var == 'NEE':
                myvar[ens[i]] = ds[str(var)][:,41:,:]
                time = ds['time']
                ## NBP < 0 = carbon source
                ## NEE > 0 = carbon source
                ## convert gC/m2/s to gC/m2/yr
                ## (unit/s * s/day * day/month) = unit/month
                ## sum over year to get unit/year
                myvarAnn[ens[i]] = (myvar[ens[i]] * time.dt.daysinmonth * 3600).groupby(
                                                'time.year').sum(dim='time',skipna=True)
            
            else:
                myvar[ens[i]] = ds[str(var)][:,41:,:]
                time = ds['time']
                myvarAnn[ens[i]] = myvar[ens[i]].groupby('time.year').mean(dim='time', skipna=True)
            ds.close()
    else:
        for i in range(numEns):
            ds = xr.open_dataset(datadir + '/' + str(simulation) + str(ens[i]) +
                                 '.cam.h0.' + str(var) + '.' + 
                                 str(timePeriod) + '_NH.nc', decode_times=False)
            units, reference_date = ds.time.attrs['units'].split('since')
            ds['time'] = pd.date_range(start=reference_date, periods=ds.sizes['time'], freq='MS')
            ## restrict to poleward of 40N ##
            lat = ds.lat[41:]
            lon = ds.lon
            myvar[ens[i]] = ds[str(var)][:,41:,:]
            myvarAnn[ens[i]] = myvar[ens[i]].groupby('time.year').mean(dim='time', skipna=True)
        ds.close()
        
    return lat, lon, myvar, myvarAnn, ens
    

def getLandType(var,warmPfrost,makeFigures):
    import xarray as xr
    import numpy as np
        
    dataDirCESM = '/Users/arielmor/Desktop/SAI/data/CESM2'
    
    ens = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010']
    numEns = len(ens)
    
    ds = xr.open_dataset(dataDirCESM + '/surfdata_0.9x1.25_hist_78pfts_CMIP6_simyr1850_c190214.nc')
    peatland = ds.peatf[107:,:] * 100.
    peatland = peatland.where(peatland > 0.)
    ds.close()
    
    ''' Grid cell area '''
    ds = xr.open_dataset(dataDirCESM + '/gridareaNH.nc')
    gridArea = ds.cell_area
    ds.close()
    
    ''' Get total peatland area in sq km (>5% peatland in grid cell) '''
    peatlandArea = np.array(np.nansum((gridArea.where(peatland.values >= 0.1)/(1000**2)),axis=(0,1)))
    print("Area of grid cells poleward of 10N that are >= 10% peatland: ", 
          np.round(peatlandArea/1e6, decimals=2), "million km2")
    return peatland

            
def findPermafrost(simulation, timePeriod):
    import numpy as np
    import xarray as xr
    import pandas as pd
    
    datadir = '/Users/arielmor/Desktop/SAI/data/ARISE/data'
    
    ens = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010']
    
    numEns       = len(ens)
    alt          = {}
    altMasked    = {}
    altAnnMean   = {}
    altmax       = {}
    altmaxMasked = {}
    altmaxAnn    = {}
    
    ds = xr.open_dataset(datadir + 
                               '/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001.clm2.h0.ALT.203501-206912_NH.nc')
    nbedrock = ds.nbedrock[41:,:]
    nbedrock = nbedrock.fillna(19)
    landmask = ds.landmask[41:,:]
    levgrnd  = [0.02,0.06,0.12,0.2,0.32,0.48,0.68,0.92,1.2,1.52,1.88,2.28,2.72,3.26,3.9,
                4.64,5.48,6.42,7.46,8.6,10.99,15.666,23.301,34.441,49.556]#ds.levgrnd
    if len(ds.lat) > 50: lat = ds.lat[41:]
    else: lat = ds.lat
    lon = ds.lon
    ds.close()
    
    
    #### index of shallowest bedrock layer
    bedrock = np.zeros((len(lat),len(lon)))
    for ilat in range(len(lat)):
        for ilon in range(len(lon)):
            bedrockIndex = int(nbedrock[ilat,ilon].values)
            bedrock[ilat,ilon] = levgrnd[bedrockIndex]
            
    
    
    if timePeriod == '201501-206912':
        bedrock = np.repeat(bedrock[None,...],660,axis=0)  
    elif timePeriod == '203501-206912':
        bedrock = np.repeat(bedrock[None,...],420,axis=0)
            
    #### constrain permafrost to only soil
    for i in range(numEns):
        ## alt
        ds = xr.open_dataset(datadir + '/' + str(simulation) + str(ens[i]) +
                             '.clm2.h0.ALT.' + 
                             str(timePeriod) + '_NH.nc', decode_times=False)
        units, reference_date = ds.time.attrs['units'].split('since')
        ds['time'] = pd.date_range(start=reference_date, periods=ds.sizes['time'], freq='MS')
        ## restrict to poleward of 50N ##
        alt[ens[i]] = ds.ALT[:,41:,:]
        ds.close()
        ## restrict to soil layers in CLM5
        altMasked[ens[i]] = alt[ens[i]].where(alt[ens[i]] <= bedrock)
        ## top 20 layers are soil - deepest bedrock starts at 8.6m, cut off at 8.60
        altMasked[ens[i]] = altMasked[ens[i]].where(altMasked[ens[i]] <= 8.60)
        ## get annual mean active layer depth
        altAnnMean[ens[i]] = altMasked[ens[i]].groupby('time.year').mean(dim='time', skipna = True)
        
    for i in range(numEns):
        ## altmax
        ds = xr.open_dataset(datadir + '/' + str(simulation) + str(ens[i]) +
                             '.clm2.h0.ALTMAX.' + 
                             str(timePeriod) + '_NH.nc', decode_times=False)
        units, reference_date = ds.time.attrs['units'].split('since')
        ds['time'] = pd.date_range(start=reference_date, periods=ds.sizes['time'], freq='MS')
        ## restrict to poleward of 50N ##
        altmax[ens[i]] = ds.ALTMAX[:,41:,:]
        ## restrict to soil layers in CLM5
        altmaxMasked[ens[i]] = altmax[ens[i]].where(altmax[ens[i]] <= bedrock)
        ## top 20 layers are soil - deepest bedrock starts at 8.6m, cut off at 8.60
        altmaxMasked[ens[i]] = altmaxMasked[ens[i]].where(altmaxMasked[ens[i]] <= 8.60)
        ## get annual maximum active layer depth
        altmaxAnn[ens[i]] = altmaxMasked[ens[i]].groupby('time.year').max(dim='time', skipna = True)
        
    return altAnnMean,altmaxMasked,altmaxAnn,lat,lon


def permafrostVolume():
    import numpy as np
    import xarray as xr
       
    datadir = '/Users/arielmor/Desktop/SAI/data/ARISE/data'
    
    ens = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010']
    
    ds = xr.open_dataset('/Users/arielmor/Desktop/SAI/data/CESM2/gridareaNH.nc')
    gridArea = ds.cell_area[41:,:]
    ds.close()
    gridAreaCONTROL = np.repeat(gridArea.values[None,...],55,axis=0)
    gridAreaFEEDBACK = np.repeat(gridArea.values[None,...],35,axis=0)
    
    ## bedrock
    ds = xr.open_dataset(datadir + 
                               '/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001.clm2.h0.ALT.203501-206912_NH.nc')
    nbedrock = ds.nbedrock[41:,:]
    nbedrock = nbedrock.fillna(24)
    levgrnd  = [0.02,0.06,0.12,0.2,0.32,0.48,0.68,0.92,1.2,1.52,1.88,2.28,2.72,3.26,3.9,
                4.64,5.48,6.42,7.46,8.6,10.99,15.666,23.301,34.441,49.556]
    if len(ds.lat) > 50: lat = ds.lat[41:]
    else: lat = ds.lat
    lon = ds.lon
    ds.close()
    
    #### index of shallowest bedrock layer
    bedrock = np.zeros((len(lat),len(lon)))
    for ilat in range(len(lat)):
        for ilon in range(len(lon)):
            bedrockIndex = int(nbedrock[ilat,ilon].values)
            bedrock[ilat,ilon] = levgrnd[bedrockIndex]
    

    bedrockCONTROL = np.repeat(bedrock[None,...],55,axis=0)  
    bedrockFEEDBACK = np.repeat(bedrock[None,...],35,axis=0)
        
    #### Annual mean permafrost volume
    from tippingPoints import findPermafrost
    altAnnualMeanCONTROL,altmaxMonthlyCONTROL,altmaxAnnCONTROL,lat,lon = findPermafrost(
        'b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.','201501-206912')
    altAnnualMeanFEEDBACK,altmaxMonthlyFEEDBACK,altmaxAnnFEEDBACK,lat,lon = findPermafrost(
        'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.', '203501-206912')
    
    pfrostVolumeAnnCONTROL  = {}
    pfrostVolumeAnnFEEDBACK = {}
    bedrock[bedrock > 8.6] = 8.6
    for ensNum in range(len(ens)):
        pfrostVolumeAnnCONTROL[ens[ensNum]]  = ((bedrockCONTROL - altAnnualMeanCONTROL[ens[ensNum]]) * gridAreaCONTROL) / (1000**3) #km3
        pfrostVolumeAnnFEEDBACK[ens[ensNum]] = ((bedrockFEEDBACK - altAnnualMeanFEEDBACK[ens[ensNum]]) * gridAreaFEEDBACK) / (1000**3) #km3
    
    
    from make_timeseries import make_timeseries, make_ensemble_mean_timeseries
    ensMembersPfrostVolumeARISE_ts = make_timeseries(10,'ALT',lat,lon,91,50,360,-1,pfrostVolumeAnnFEEDBACK)
    ensMembersPfrostVolumeSSP_ts   = make_timeseries(10,'ALT',lat,lon,91,50,360,-1,pfrostVolumeAnnCONTROL)
    
    ensMeanPfrostVolumeARISE_ts = make_ensemble_mean_timeseries(ensMembersPfrostVolumeARISE_ts, 10)
    ensMeanPfrostVolumeSSP_ts   = make_ensemble_mean_timeseries(ensMembersPfrostVolumeSSP_ts, 10)
    
    return ensMembersPfrostVolumeARISE_ts,ensMembersPfrostVolumeSSP_ts,ensMeanPfrostVolumeARISE_ts,ensMeanPfrostVolumeSSP_ts


def permafrostTemperature(simulation, timePeriod):
    import numpy as np
    import xarray as xr
    import pandas as pd
       
    datadir = '/Users/arielmor/Desktop/SAI/data/ARISE/data'
    ens    = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010']
    numEns = len(ens)
    
    #### Get soil column temperature
    
    alt           = {}
    altMasked     = {}
    altAnn        = {}
    altAnnMean    = {}
    tsoi          = {}
    tsoiAnnMean   = {}
    
    ds = xr.open_dataset(datadir + 
                         '/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001.clm2.h0.ALT.203501-206912_NH.nc')
    nbedrock = ds.nbedrock[41:,:]
    nbedrock = nbedrock.fillna(19)
    levgrnd  = [0.02,0.06,0.12,0.2,0.32,0.48,0.68,0.92,1.2,1.52,1.88,2.28,2.72,3.26,3.9,
                4.64,5.48,6.42,7.46,8.6,10.99,15.666,23.301,34.441,49.556]
    # soil thickness for weighting
    dzsoi    = [0.02,0.04,0.06,0.06,0.12,0.16,0.2,0.24,0.28,0.32,0.36,0.4,0.44,0.54,0.64,
                0.74,0.84,0.94,1.04,1.14,2.39,4.676,7.635,11.14,15.115]
    if len(ds.lat) > 50: lat = ds.lat[41:]
    else: lat = ds.lat
    lon = ds.lon
    ds.close()
    
    #### index of shallowest bedrock layer
    bedrock = np.zeros((len(lat),len(lon)))
    for ilat in range(len(lat)):
        for ilon in range(len(lon)):
            bedrockIndex = int(nbedrock[ilat,ilon].values)
            bedrock[ilat,ilon] = levgrnd[bedrockIndex]
    

    #### constrain permafrost to only soil
    for i in range(numEns):
        ds = xr.open_dataset(datadir + '/' + str(simulation) + str(ens[i]) +
                              '.clm2.h0.ALT.' + 
                              str(timePeriod) + '_NH.nc', decode_times=False)
        units, reference_date = ds.time.attrs['units'].split('since')
        ds['time'] = pd.date_range(start=reference_date, periods=ds.sizes['time'], freq='MS')
        ## restrict to poleward of 50N ##
        alt[ens[i]] = ds.ALT[:,41:,:]
        ## restrict to soil layers in CLM5
        altMasked[ens[i]] = alt[ens[i]].where(alt[ens[i]] <= bedrock)
        ## top 20 layers are soil - deepest bedrock starts at 8.6m, cut off at 8.60
        altMasked[ens[i]] = altMasked[ens[i]].where(altMasked[ens[i]] <= 8.60)
        altAnn[ens[i]] = altMasked[ens[i]].groupby('time.year').max(dim='time', skipna = True)
        altAnnMean[ens[i]] = altMasked[ens[i]].groupby('time.year').mean(dim='time', skipna = True)
        
        #### soil temperature
        ds = xr.open_dataset(datadir + '/' + str(simulation) + str(ens[i]) +
                              '.clm2.h0.TSOI.' + 
                              str(timePeriod) + '_NH.nc', decode_times=False)
        units, reference_date = ds.time.attrs['units'].split('since')
        ds['time'] = pd.date_range(start=reference_date, periods=ds.sizes['time'], freq='MS')
        tsoi[ens[i]] = ds.TSOI[:,:21,41:,:] - 273.15
        tsoi[ens[i]] = tsoi[ens[i]].where(altMasked[ens[i]] <= 8.60)
        tsoiAnnMean[ens[i]] = tsoi[ens[i]].groupby('time.year').mean(dim='time', skipna = True)
        
    levmin_ind = {}
    for i in range(numEns):
        print(i)
        levmin_ind[ens[i]] = np.zeros((altAnnMean[ens[0]].shape[0], len(lat), len(lon)))
        for iyear in range(altAnnMean[ens[0]].shape[0]):
            for ilat in range(len(lat)):
                for ilon in range(len(lon)):
                    #### find soil level index for bottom of active layer
                    ## weighted average from bottom of active layer to top of bedrock
                    levmin_ind[ens[i]][iyear,ilat,ilon] = int(np.abs(
                            np.array(levgrnd) - altAnnMean[ens[i]][iyear,ilat,ilon].values).argmin())
                    
     
    levmax_ind = nbedrock.values
    levmax_ind[levmax_ind > 19] = 19
    #### weighted average permafrost temperature
    ## levmin_ind = soil level index of bottom of active layer
    ## levmax_ind = soil level index of top of bedrock
    weighted_avg = {}
    for i in range(numEns):
        print(i)
        weighted_avg[ens[i]] = np.zeros((altAnnMean[ens[0]].shape[0], len(lat), len(lon)))
        for iyear in range(altAnnMean[ens[0]].shape[0]):
            for ilat in range(len(lat)):
                for ilon in range(len(lon)):
                    if levmin_ind[ens[i]][iyear,ilat,ilon] == (levmax_ind[ilat,ilon]): 
                        levmax_ind[ilat,ilon] = (levmax_ind[ilat,ilon])+1
                    tsoiPfrost = tsoiAnnMean[ens[i]][iyear,int(levmin_ind[ens[i]][iyear,ilat,ilon]):int(levmax_ind[ilat,ilon]),ilat,ilon]
                    tsoiPfrost = tsoiPfrost.fillna(0)
                    weighted_avg[ens[i]][iyear,ilat,ilon] = np.average(tsoiPfrost, weights=dzsoi[int(levmin_ind[ens[i]][iyear,ilat,ilon]):int(levmax_ind[ilat,ilon])])
    
    weighted_avg_masked = {}
    for i in range(numEns):
        weighted_avg_masked[ens[i]] = np.where(altAnnMean[ens[i]] <=8.6, weighted_avg[ens[i]], np.nan)
    
    if timePeriod == '203501-206912':
        np.save(datadir + '/permafrost_temperature_FEEDBACK.npy', weighted_avg_masked)
    elif timePeriod == '201501-206912':
        np.save(datadir + '/permafrost_temperature_CONTROL.npy', weighted_avg_masked)
        
        
    #### integrated temperature down to bedrock
    integral = {}
    for i in range(numEns):
        print(i)
        integral[ens[i]] = np.zeros((altAnnMean[ens[0]].shape[0], len(lat), len(lon)))
        for iyear in range(altAnnMean[ens[0]].shape[0]):
            for ilat in range(len(lat)):
                for ilon in range(len(lon)):
                    # if levmin_ind[ens[i]][iyear,ilat,ilon] == (levmax_ind[ilat,ilon]): 
                    #     levmax_ind[ilat,ilon] = (levmax_ind[ilat,ilon])+1
                    tsoiSoil = tsoiAnnMean[ens[i]][iyear,:int(levmax_ind[ilat,ilon]),ilat,ilon]
                    tsoiSoil = tsoiSoil.fillna(0)
                    integral[ens[i]][iyear,ilat,ilon] = np.trapz(tsoiSoil, axis=0)
    
    integral_masked = {}
    for i in range(numEns):
        integral_masked[ens[i]] = np.where(altAnnMean[ens[i]] <=8.6, integral[ens[i]], np.nan)
    
    import pickle
    if timePeriod == '203501-206912':
        with open('/Users/arielmor/Desktop/SAI/data/ARISE/data/integrated_soil_temperature_to_bedrock_FEEDBACK.pkl', 'wb') as fp:
            pickle.dump(integral_masked, fp)
            print('Your dictionary has been saved successfully to file')
    elif timePeriod == '201501-206912':
        with open('/Users/arielmor/Desktop/SAI/data/ARISE/data/integrated_soil_temperature_to_bedrock_CONTROL.pkl', 'wb') as fp:
            pickle.dump(integral_masked, fp)
            print('Your dictionary has been saved successfully to file')
    
    return weighted_avg_masked,integral_masked


def findTippingPoint(warmPfrost):
    import numpy as np
    import xarray as xr
    from tippingPoints import readDataForTippingPoints, findThawDate, talikFormation, findPermafrost
    from peatlandPermafrostCalculations import getLandType
    from make_timeseries import make_ensemble_mean_timeseries
    import cartopy.crs as ccrs
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    import matplotlib.ticker as mticker
    from cartopy.util import add_cyclic_point
    import matplotlib.pyplot as plt
    import matplotlib.path as mpath
    import matplotlib.colors as mcolors
    from plottingFunctions import get_colormap, circleBoundary, mapsSubplotsDiff, mapsSubplots
    from matplotlib import colors as c
    vfont = {'fontname':'Verdana'}
    hfont = {'fontname':'Helvetica'}
    circle = circleBoundary
    brbg_cmap,rdbu_cmap,jet,magma,reds,hot,seismic = get_colormap(27)
    
    datadir = '/Users/arielmor/Desktop/SAI/data/ARISE/data'
    
    ens = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010']
    numEns = len(ens)
    
    ## landmask ##
    ds = xr.open_dataset(datadir + '/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001.clm2.h0.ALT.203501-206912_NH.nc')
    landmask = ds.landmask
    ds.close()
    
    landMask = landmask.where(np.isnan(landmask))
    landMask = landmask.copy() + 2
    landMask = np.where(~np.isnan(landmask),landMask, 1)    
    cmapLand = c.ListedColormap(['xkcd:gray','none'])
    
    
    ''' Read data '''  
    ########################################################
    ####     Pfrost extent (active layer)     ####
    ########################################################
    altAnnMeanCONTROL,pfrostCONTROL,pfrostAnnCONTROL,lat,lon = findPermafrost(
        'b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.', '201501-206912')
    altAnnMeanFEEDBACK,pfrostFEEDBACK,pfrostAnnFEEDBACK,lat,lon = findPermafrost(
        'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.', '203501-206912')
    
    
    pfrostExtentCONTROL = {}
    pfrostExtentFEEDBACK = {}
    for numEns in range(len(ens)):
        '''ssp'''
        pfrostExtentCONTROL[ens[numEns]] = pfrostAnnCONTROL[ens[numEns]].copy()
        pfrostExtentCONTROL[ens[numEns]] = xr.where(pfrostAnnCONTROL[ens[numEns]][0,:,:].notnull(), 1, 0)
        '''default arise'''
        pfrostExtentFEEDBACK[ens[numEns]] = pfrostAnnFEEDBACK[ens[numEns]].copy()
        pfrostExtentFEEDBACK[ens[numEns]] = xr.where(pfrostAnnFEEDBACK[ens[numEns]][0,:,:].notnull(), 1, 0)
    
    
    #########################################################
    #### Total column soil temp - for talik formation
    #########################################################
    lat, lon, tsCONTROL, tsAnnCONTROL, ens = readDataForTippingPoints(
        'TSOI', False, False, 'b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.', '201501-206912', 
                            True, warmPfrost)
    lat, lon, tsFEEDBACK, tsAnnFEEDBACK, ensFEEDBACK = readDataForTippingPoints(
        'TSOI', False, False, 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.', '203501-206912', 
                            True, warmPfrost)
    
    ## can only have talik where permafrost exists
    for ensNum in range(len(ens)):
        mask = np.repeat(pfrostCONTROL[ens[ensNum]].values[:,np.newaxis,:,:],20,axis=1)
        tsCONTROL[ens[ensNum]] = xr.where(~np.isnan(mask), tsCONTROL[ens[ensNum]], np.nan)
        del mask
        mask = np.repeat(pfrostAnnCONTROL[ens[ensNum]].values[:,np.newaxis,:,:],20,axis=1)
        tsAnnCONTROL[ens[ensNum]] = xr.where(~np.isnan(mask), tsAnnCONTROL[ens[ensNum]], np.nan)
        del mask
        
    
    for ensNum in range(len(ens)):
        mask = np.repeat(pfrostFEEDBACK[ens[ensNum]].values[:,np.newaxis,:,:],20,axis=1)
        tsFEEDBACK[ens[ensNum]] = xr.where(~np.isnan(mask), tsFEEDBACK[ens[ensNum]], np.nan)
        del mask
        mask = np.repeat(pfrostAnnFEEDBACK[ens[ensNum]].values[:,np.newaxis,:,:],20,axis=1)
        tsAnnFEEDBACK[ens[ensNum]] = xr.where(~np.isnan(mask), tsAnnFEEDBACK[ens[ensNum]], np.nan)
        del mask
    
    
    tsAnnCONTROLmean  = np.nanmean(np.stack((tsAnnCONTROL.values())),axis=0) 
    tsAnnFEEDBACKmean = np.nanmean(np.stack((tsAnnFEEDBACK.values())),axis=0)
                
    
    #### Talik formation timing
    ''' ------------------------------------------------------ '''
    ''' Find talik formation timing based on column soil temp  '''
    ''' ------------------------------------------------------ '''
    #########################################################
    ## CONTROL TALIK FORMATION
    #########################################################
    ds = xr.open_dataset(datadir + 
                               '/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001.clm2.h0.ALT.203501-206912_NH.nc')
    nbedrock = ds.nbedrock[41:,:]
    nbedrock = nbedrock.fillna(20)
    ds.close()
    
    numEns = len(ens)
    talikAnnCONTROL = {}
    for numEns in range(len(ens)):
        print(numEns)
        talikAnnCONTROL[ens[numEns]] = np.empty((tsAnnCONTROLmean.shape[1]-1,len(lat),len(lon)))*np.nan
        for ilat in range(len(lat)):
            for ilon in range(len(lon)):
                for ilev in range(int(nbedrock[ilat,ilon])-1):
                # for ilev in range(tsAnnCONTROLmean.shape[1]-1):
                    talikAnnCONTROL[ens[numEns]][ilev,ilat,ilon] = talikFormation((tsAnnCONTROL[ens[numEns]][:,ilev+1,ilat,ilon]),N=2)
        talikAnnCONTROL[ens[numEns]] = np.nanmin(talikAnnCONTROL[ens[numEns]], axis=0)
    import pickle
    with open('/Users/arielmor/Desktop/SAI/data/ARISE/data/talikAnnCONTROL.pkl', 'wb') as fp:
        pickle.dump(talikAnnCONTROL, fp)
        print('Your dictionary has been saved successfully to file')
    
                
    talikAnnCONTROLmean = np.empty((tsAnnCONTROLmean.shape[1]-1,len(lat),len(lon)))*np.nan
    for ilat in range(len(lat)):
        for ilon in range(len(lon)):
            for ilev in range(int(nbedrock[ilat,ilon])-1):
            # for ilev in range(tsAnnCONTROLmean.shape[1]-1):
                talikAnnCONTROLmean[ilev,ilat,ilon] = talikFormation((tsAnnCONTROLmean[:,ilev+1,ilat,ilon]),N=2)
    talikAnnCONTROLmean = np.nanmin(talikAnnCONTROLmean,axis=0)
    
                
    
    #########################################################
    ## FEEDBACK TALIK FORMATION
    #########################################################
    numEns = len(ens)
    talikAnnFEEDBACK = {}
    for numEns in range(len(ens)):
        print(numEns)
        tsAnnFEEDBACK_combined = np.concatenate((tsAnnCONTROL[ens[numEns]][:20,:,:,:],tsAnnFEEDBACK[ens[numEns]]),axis=0)
        talikAnnFEEDBACK[ens[numEns]] = np.empty((tsAnnFEEDBACKmean.shape[1]-1,len(lat),len(lon)))*np.nan
        for ilat in range(len(lat)):
            for ilon in range(len(lon)):
                for ilev in range(int(nbedrock[ilat,ilon])-1):
                # for ilev in range(tsAnnFEEDBACK[ensFEEDBACK[0]].shape[1]-1):
                    talikAnnFEEDBACK[ens[numEns]][ilev,ilat,ilon] = talikFormation((tsAnnFEEDBACK_combined[:,ilev+1,ilat,ilon]),N=2)
        del tsAnnFEEDBACK_combined
        talikAnnFEEDBACK[ensFEEDBACK[numEns]] = np.nanmin(talikAnnFEEDBACK[ens[numEns]], axis=0)
        
    import pickle
    with open('/Users/arielmor/Desktop/SAI/data/ARISE/data/talikAnnFEEDBACK.pkl', 'wb') as fp:
        pickle.dump(talikAnnFEEDBACK, fp)
        print('Your dictionary has been saved successfully to file')
                
    
    tsAnnFEEDBACK_combined_mean = np.concatenate((tsAnnCONTROLmean[:20,:,:,:],tsAnnFEEDBACKmean),axis=0)
    talikAnnFEEDBACKmean = (np.empty((tsAnnFEEDBACKmean.shape[1]-1,len(lat),len(lon))))*np.nan
    for ilat in range(len(lat)):
        for ilon in range(len(lon)):
            for ilev in range(int(nbedrock[ilat,ilon])-1):
            # for ilev in range(tsAnnFEEDBACKmean.shape[1]-1):
                talikAnnFEEDBACKmean[ilev,ilat,ilon] = talikFormation((tsAnnFEEDBACK_combined_mean[:,ilev+1,ilat,ilon]),N=2)
    talikAnnFEEDBACKmean = np.nanmin(talikAnnFEEDBACKmean,axis=0)
    
    
    np.save(datadir + '/talikAnnCONTROL_NEW.npy', talikAnnCONTROL)
    np.save(datadir + '/talikAnnCONTROLmean.npy', talikAnnCONTROLmean)
    np.save(datadir + '/talikAnnFEEDBACK_NEW.npy', talikAnnFEEDBACK)
    np.save(datadir + '/talikAnnFEEDBACKmean.npy', talikAnnFEEDBACKmean)
    
    
    '''from 2035'''
    numEns = len(ens)
    talikAnnCONTROL_2035 = {}
    for numEns in range(len(ens)):
        print(numEns)
        talikAnnCONTROL_2035[ens[numEns]] = np.zeros(((tsAnnCONTROLmean[20:,:,:,:].shape[1]-1),len(lat),len(lon)))
        for ilat in range(len(lat)):
            for ilon in range(len(lon)):
                for ilev in range(tsAnnCONTROLmean[20:,:,:,:].shape[1]-1):
                    talikAnnCONTROL_2035[ens[numEns]][ilev,ilat,ilon] = talikFormation((tsAnnCONTROL[ens[numEns]][20:,ilev+1,ilat,ilon]),N=2)
        talikAnnCONTROL_2035[ens[numEns]] = np.nanmin(talikAnnCONTROL_2035[ens[numEns]], axis=0)
        np.save('talikAnnSoilOnlyCONTROL_ens' + str(numEns + 1) + '_from_2035.npy', talikAnnCONTROL_2035[ens[numEns]])
                
    talikAnnCONTROLmean_2035 = np.zeros(((tsAnnCONTROLmean[20:,:,:,:].shape[1]-1),len(lat),len(lon)))
    for ilat in range(len(lat)):
        for ilon in range(len(lon)):
            for ilev in range(tsAnnCONTROLmean[20:,:,:,:].shape[1]-1):
                talikAnnCONTROLmean_2035[ilev,ilat,ilon] = talikFormation((tsAnnCONTROLmean[20:,ilev+1,ilat,ilon]),N=2)
    talikAnnCONTROLmean_2035 = np.nanmin(talikAnnCONTROLmean_2035,axis=0)
    
    
    ''' ------------------------- '''      
    #### Peatland extent       
    ''' ------------------------- ''' 
    #########################################################
    # Peatland fraction (at least 10%)
    #########################################################
    peatland = getLandType('ER',False,False)
    peatlandContour = peatland.copy(); peatlandContour = xr.where(peatland >= 10, 1, 0)
        
        
    ''' ------------------------------ '''
    #### FIGURES: talik
    ''' ------------------------------ '''
    #########################################################
    #### Talik: ens mean control
    #########################################################
    longitude = lon
    var,lon2 = add_cyclic_point(talikAnnCONTROLmean,coord=longitude)
    hfont = {'fontname':'Calibri'}
    
    ## Create figure
    fig = plt.figure(figsize=(10,6))
    norm = mcolors.Normalize(vmin=0, vmax=54)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
    ax.set_extent([180, -180, 50, 90], crs=ccrs.PlateCarree())
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.set_facecolor('0.8')
    
    ## Filled contour map
    cmapLand = c.ListedColormap(['xkcd:gray','none'])
    ax.pcolormesh(lon,lat,landMask[41:,:],transform=ccrs.PlateCarree(),cmap=cmapLand)
    cf1 = ax.pcolormesh(lon2,lat,var,transform=ccrs.PlateCarree(), 
                  norm=norm, cmap=magma)
    ax.contour(lon,lat,peatlandContour[41:,:],[1],colors='g',
                    linewidth=0.4,transform=ccrs.PlateCarree())
    ## use below for contour line of pfrost extent in control
    # ax.contour(lon,lat,pfrostExtentCONTROLmean[41:,:],[1],colors='b',
    #             linewidth=0.5,transform=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.9)
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), 
                      draw_labels=True,
                      linewidth=1, color='C7', 
                  alpha=0.8, linestyle=':',
                  x_inline=False,
                  y_inline=True,
                  rotate_labels=False)
    gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120])
    gl.ylocator = mticker.FixedLocator([60, 70, 80])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.ylabel_style = {'size': 9, 'rotation':20}
    plt.draw()  # Enable the use of `gl._labels`
    for ea in gl.label_artists:
        # return ea of mpl.text type, e.g. Text(135, 30, '30°N') 
        pos = ea.get_position()
        if pos[0] == 150:
            ea.set_position([0, pos[1]])
            
    cbar = plt.colorbar(cf1, ax=ax, ticks=[0,10,20,30,40,50],fraction=0.046)
    cbar.ax.set_yticklabels(['2015','2025','2035','2045','2055','2065'], **hfont)
    cbar.set_label('Talik formation year', fontsize=11, **hfont)
    plt.title('a) SSP2-4.5 talik formation year', fontsize=12, fontweight='bold', **hfont)
    ## Save figure
    plt.savefig('/Users/arielmor/Desktop/SAI/data/ARISE/figures/control_talik_formation_year_soil_only_MEAN_peat.jpg', 
                dpi=1200, bbox_inches='tight')
    del fig,ax,var,lon2,longitude,cf1,cbar
    
    
    #########################################################
    #### Talik: ens mean feedback
    #########################################################
    longitude = lon
    var,lon2 = add_cyclic_point(talikAnnFEEDBACKmean,coord=longitude)
    
    ## Create figure
    fig = plt.figure(figsize=(10,6))
    norm = mcolors.Normalize(vmin=0, vmax=54)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
    ax.set_extent([180, -180, 50, 90], crs=ccrs.PlateCarree())
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.set_facecolor('0.8')
    ## Filled contour map
    ax.pcolormesh(lon,lat,landMask[41:,:],transform=ccrs.PlateCarree(),cmap=cmapLand)
    cf1 = ax.pcolormesh(lon2,lat,var,transform=ccrs.PlateCarree(), 
                  norm=norm, cmap=magma)
    ax.contour(lon,lat,peatlandContour[41:,:],[1],colors='g',
                    linewidth=0.4,transform=ccrs.PlateCarree())
    # ax.contour(lon,lat,pfrostExtentCONTROLmean[41:,:],[1],colors='b',
    #             linewidth=0.5,transform=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.9)
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), 
                      draw_labels=True,
                      linewidth=1, color='C7', 
                  alpha=0.8, linestyle=':',
                  x_inline=False,
                  y_inline=True,
                  rotate_labels=False)
    gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120])
    gl.ylocator = mticker.FixedLocator([60, 70, 80])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    #gl.xlabel_style = {'rotation':0}
    gl.ylabel_style = {'size': 9, 'rotation':20}
    plt.draw()  # Enable the use of `gl._labels`
    for ea in gl.label_artists:
        # return ea of mpl.text type, e.g. Text(135, 30, '30°N') 
        pos = ea.get_position()
        if pos[0] == 150:
            ea.set_position([0, pos[1]])
    
    cbar = plt.colorbar(cf1, ax=ax, ticks=[0,10,20,30,40,50],fraction=0.046)
    # cbar = plt.colorbar(cf1, ax=ax, ticks=[0,10,20,30,40,50],fraction=0.045,location='bottom',orientation='horizontal')
    cbar.ax.set_yticklabels(['2015','2025','2035','2045','2055','2065'], **hfont)
    cbar.set_label('Talik formation year', fontsize=11, **hfont)
    plt.title('b) ARISE-SAI-1.5 talik formation year', fontsize=12, fontweight='bold', **hfont)
    ## Save figure
    plt.savefig('/Users/arielmor/Desktop/SAI/data/ARISE/figures/feedback_talik_formation_year_soil_only_MEAN_peat.jpg', 
                dpi=1200, bbox_inches='tight')
    del fig,ax,var,lon2,longitude,cf1,cbar
    
    
    #########################################################
    #### Talik: diff
    #########################################################
    talikAnnFEEDBACKmean = np.load(datadir + '/talikAnnFEEDBACKmean.npy')
    talikAnnCONTROLmean = np.load(datadir + '/talikAnnCONTROLmean.npy')
    cMapthawedControlNotFeedback = c.ListedColormap(['xkcd:aqua blue'])
    cMapthawedFeedbackNotControl = c.ListedColormap(['xkcd:bright yellow'])
    cMapALWAYSTHAW = c.ListedColormap(['k'])
    
    
    ## thawed in control but not in feedback
    thawedControlNotFeedback = talikAnnFEEDBACKmean.copy()
    thawedControlNotFeedback[
        (np.isnan(talikAnnFEEDBACKmean)) & 
        (~np.isnan(talikAnnCONTROLmean))] = 100 # nan in feedback mean = didn't thaw in FB
    thawedControlNotFeedback[thawedControlNotFeedback < 100.] = np.nan
    
    ## talik in feedback but not in control:
    thawedFeedbackNotControl = talikAnnCONTROLmean.copy()
    thawedFeedbackNotControl[
        (np.isnan(talikAnnCONTROLmean)) & 
        (~np.isnan(talikAnnFEEDBACKmean))] = -100
    thawedFeedbackNotControl[thawedFeedbackNotControl > -100.] = np.nan
    
    ## thawed by 2035:
    talikALWAYSmean = talikAnnCONTROLmean.copy()
    talikALWAYSmean[
        (talikAnnCONTROLmean >= 20) |
        (talikAnnFEEDBACKmean >= 20)] = np.nan
    
            
    ## difference in thaw timing between control and feedback
    diffMEAN = talikAnnCONTROLmean - talikAnnFEEDBACKmean
    
    ## mask out cells that didn't thaw in control or feedback (diff = nan) and always thawed
    diffMEAN[(thawedControlNotFeedback == 100) | (thawedFeedbackNotControl == -100) | (talikALWAYSmean < 20)] = np.nan # == 0

    
    ## Create figure ##
    fig = plt.figure(figsize=(10,6))
    norm = mcolors.TwoSlopeNorm(vmin=-10, vcenter=0, vmax=10)
    longitude = lon; plottingVarMean,lon2 = add_cyclic_point(diffMEAN,coord=longitude)
    
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
    ax.set_extent([180, -180, 50, 90], crs=ccrs.PlateCarree())
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes); ax.set_facecolor('0.8')
    
    ## land mask
    ax.pcolormesh(lon,lat,landMask[41:,:],transform=ccrs.PlateCarree(),cmap=cmapLand)
    ## thawed in control but not feedback
    ax.pcolormesh(lon,lat,thawedControlNotFeedback,transform=ccrs.PlateCarree(),
                        cmap=cMapthawedControlNotFeedback)
    ## thawed in feedback but not control
    ax.pcolormesh(lon,lat,thawedFeedbackNotControl,transform=ccrs.PlateCarree(),
                        cmap=cMapthawedFeedbackNotControl)
    ## always thawed = black
    ax.pcolormesh(lon,lat,talikALWAYSmean,transform=ccrs.PlateCarree(),
                        cmap=cMapALWAYSTHAW)
    ## difference in thaw timing
    cf1 = ax.pcolormesh(lon2,lat,plottingVarMean,transform=ccrs.PlateCarree(), 
                  norm=norm, cmap='bwr') # seismic
    ## peatland contour
    ax.contour(lon,lat,peatlandContour[41:,:],[1],colors='g',
                    linewidth=0.5,transform=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.9)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), 
                      draw_labels=True,
                      linewidth=1, color='C7', 
                  alpha=0.8, linestyle=':',
                  x_inline=False,
                  y_inline=True,
                  rotate_labels=False)
    gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120])
    gl.ylocator = mticker.FixedLocator([60, 70, 80])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    #gl.xlabel_style = {'rotation':0}
    gl.ylabel_style = {'size': 9, 'rotation':20}
    plt.draw()  # Enable the use of `gl._labels`
    for ea in gl.label_artists:
        # return ea of mpl.text type, e.g. Text(135, 30, '30°N') 
        pos = ea.get_position()
        if pos[0] == 150:
            ea.set_position([0, pos[1]])
    cbar = plt.colorbar(cf1, ax=ax, ticks=[-8.5, 0, 8.5], fraction=0.046)
    # cbar = plt.colorbar(cf1, ax=ax, ticks=[-8.5, 0, 8.5], fraction=0.045,location='bottom',orientation='horizontal')
    cbar.ax.set_yticklabels(['Talik forms\nearlier in\nSSP2-4.5', 
                             'Talik forms\nsame year', 
                             'Talik forms\nearlier in\nARISE-SAI-1.5'])
    cbar.ax.tick_params(size=0)
    
    plt.title('c) Talik formation year, SSP2-4.5 minus ARISE-SAI', fontsize=12, fontweight='bold', **hfont)
    plt.savefig('/Users/arielmor/Desktop/SAI/data/ARISE/figures/control_minus_feedback_talik_formation_soil_only_MEAN_peat.jpg', 
                dpi=1200, bbox_inches='tight')
    del fig, ax, cf1
    
    
    '''----------------------------------------'''
    #### area of talik prevented
    ds = xr.open_dataset('/Users/arielmor/Desktop/SAI/data/CESM2/gridareaNH.nc')
    gridArea = ds.cell_area
    ds.close()
    
    ## grid areas
    print("Talik area in SSP2-4.5: ", (np.array(np.nansum(gridArea[41:,:].where(~np.isnan(talikAnnCONTROLmean))/(1000**2),axis=(0,1))))/1e6, "million km2")
    gridAreaThaw = np.array(np.nansum((gridArea[41:,:].where(
        ~np.isnan(thawedControlNotFeedback))/(1000**2)),axis=(0,1)))
    print("Talik prevented by SAI: ", np.round(gridAreaThaw/1e6, decimals=2), "million km2")
    gridAreaSAI = np.array(np.nansum((gridArea[41:,:].where(~np.isnan(thawedFeedbackNotControl))/(1000**2)),axis=(0,1)))
    print("Talik caused by SAI: ", np.round(gridAreaSAI/1e6, decimals=2), "million km2")
    gridAreaSSP = np.array(np.nansum((gridArea[41:,:].where((diffMEAN < 0))/(1000**2)),axis=(0,1)))
    print("Talik delayed by SAI: ", np.round(gridAreaSSP/1e6, decimals=2), "million km2")
    # gridAreaPeat = np.array(np.nansum((gridArea[41:,:].where(
    #     (~np.isnan(thawedControlNotFeedback)) | (diffMEAN < 0) & (
    #         peatland[41:,:].values > 10))/(1000**2)),axis=(0,1)))
    gridAreaPeat = np.array(np.nansum((gridArea[41:,:].where(
        (~np.isnan(thawedControlNotFeedback)) & (
            peatland[41:,:].values > 0))/(1000**2)),axis=(0,1)))
    print("Peat talik prevented by SAI: ", np.round(gridAreaPeat/1e6, decimals=2), "million km2")
    print("Percent talik prevented by SAI in peat: ", (gridAreaPeat/(gridAreaThaw+gridAreaSSP))*100)
    
    
    
    #### talik area time series
    ds = xr.open_dataset('/Users/arielmor/Desktop/SAI/data/CESM2/gridareaNH.nc')
    gridArea = ds.cell_area[41:,:]
    ds.close()
    
    import pickle
    with open('/Users/arielmor/Desktop/SAI/data/ARISE/data/talikAnnFEEDBACK.pkl', 'rb') as fp:
        talikAnnFEEDBACK = pickle.load(fp)
    with open('/Users/arielmor/Desktop/SAI/data/ARISE/data/talikAnnCONTROL.pkl', 'rb') as fp:
        talikAnnCONTROL = pickle.load(fp)
    
    # control
    talikAreaCONTROL  = np.zeros((55))
    talikAreaCONTROL[0] = np.nansum(gridArea.where(talikAnnCONTROLmean == 0),axis=(0,1))
    # feedback
    talikAreaFEEDBACK = np.zeros((55))
    talikAreaFEEDBACK[0] = talikAreaFEEDBACK[0] + np.nansum(
        gridArea.where(talikAnnFEEDBACKmean == 0),axis=(0,1))
    # calculation
    for iyear in range(1,55):
        talikAreaCONTROL[iyear] = talikAreaCONTROL[iyear-1] + np.nansum(
            gridArea.where(talikAnnCONTROLmean == iyear),axis=(0,1))
        talikAreaFEEDBACK[iyear] = talikAreaFEEDBACK[iyear-1] + np.nansum(
            gridArea.where(talikAnnFEEDBACKmean == iyear),axis=(0,1))
        
    ## ensemble members
    talikAreaEnsCONTROL  = {}
    talikAreaEnsFEEDBACK = {}
    for i in range(len(ens)):
        talikAreaEnsCONTROL[ens[i]]  = np.zeros((55))
        talikAreaEnsCONTROL[ens[i]][0] = np.nansum(gridArea.where(talikAnnCONTROL[ens[i]] == 0),axis=(0,1))
        talikAreaEnsFEEDBACK[ens[i]] = np.zeros((55))
        talikAreaEnsFEEDBACK[ens[i]][0] = np.nansum(gridArea.where(
                                    talikAnnFEEDBACK[ens[i]] == 0),axis=(0,1))
        for iyear in range(1,55):
            talikAreaEnsCONTROL[ens[i]][iyear] = talikAreaEnsCONTROL[ens[i]][iyear-1] + np.nansum(
                gridArea.where(talikAnnCONTROL[ens[i]] == iyear),axis=(0,1))
            talikAreaEnsFEEDBACK[ens[i]][iyear] = talikAreaEnsFEEDBACK[ens[i]][iyear-1] + np.nansum(
                gridArea.where(talikAnnFEEDBACK[ens[i]] == iyear),axis=(0,1))
            
    talikAreaCONTROL = make_ensemble_mean_timeseries(talikAreaEnsCONTROL, 10)
    talikAreaFEEDBACK = make_ensemble_mean_timeseries(talikAreaEnsFEEDBACK, 10)
        
    ## figure
    fig, ax = plt.subplots(1,1, figsize=(9,5), dpi=1200)
    for i in range(len(ens)):
        ax.plot(np.linspace(2015,2069,55),talikAreaEnsCONTROL[ens[i]]/(1000**2)/1e6,color='xkcd:light red',label='SSP2-4.5',linestyle='--')
        ax.plot(np.linspace(2035,2069,35),talikAreaEnsFEEDBACK[ens[i]][20:]/(1000**2)/1e6,color='xkcd:dark sky blue',label='ARISE-SAI-1.5')
    ax.plot(np.linspace(2015,2069,55),talikAreaCONTROL/(1000**2)/1e6,linewidth=2,color='xkcd:dark red',linestyle='--')
    ax.plot(np.linspace(2035,2069,35),talikAreaFEEDBACK[20:]/(1000**2)/1e6,linewidth=2,color='xkcd:dark blue')
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='xkcd:light red', lw=2, linestyle='--'),
                    Line2D([0], [0], color='xkcd:dark blue', lw=2)]
    plt.legend(custom_lines, ['SSP2-4.5','ARISE-SAI-1.5'], fancybox=True, fontsize=12)
    plt.xlim([2015,2069])
    #plt.ylim([0.4e6,4.8e6])
    ax.set_xticks([2015,2025,2035,2045,2055,2065])
    ax.set_xticklabels(['2015','2025','2035','2045','2055','2065'])
    # ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
    plt.ylabel('Area (km$^2$ x 10$^6$)', fontsize=12)
    plt.title('Talik area', fontsize=14, fontweight='bold')
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.savefig('/Users/arielmor/Desktop/SAI/data/ARISE/figures/talik_area_timeseries.jpg',
                dpi=1200, bbox_inches='tight')
    
    
    ## talik in peat
    gridAreaPeat = gridArea.where(peatland[41:,].values >= 10)
    
    # control
    talikAreaCONTROL_peat  = np.zeros((55))
    talikAreaCONTROL_peat[0] = np.nansum(gridAreaPeat.where(talikAnnCONTROLmean == 0),axis=(0,1))
    # feedback
    talikAreaFEEDBACK_peat = np.zeros((55))
    talikAreaFEEDBACK_peat[0] = np.nansum(gridAreaPeat.where(talikAnnFEEDBACKmean == 0),axis=(0,1))
    # calculation
    for iyear in range(1,55):
        talikAreaCONTROL_peat[iyear] = talikAreaCONTROL_peat[iyear-1] + np.nansum(
            gridAreaPeat.where(talikAnnCONTROLmean == iyear),axis=(0,1))
        talikAreaFEEDBACK_peat[iyear] = talikAreaFEEDBACK_peat[iyear-1] + np.nansum(
            gridAreaPeat.where(talikAnnFEEDBACKmean == iyear),axis=(0,1))
        
        
    ## ensemble members
    talikAreaEnsCONTROL_peat  = {}
    talikAreaEnsFEEDBACK_peat = {}
    for i in range(len(ens)):
        print(i)
        talikAreaEnsCONTROL_peat[ens[i]]  = np.zeros((55))
        talikAreaEnsCONTROL_peat[ens[i]][0] = np.nansum(gridAreaPeat.where(
                                    talikAnnCONTROL[ens[i]] == 0),axis=(0,1))
        talikAreaEnsFEEDBACK_peat[ens[i]] = np.zeros((55)) 
        talikAreaEnsFEEDBACK_peat[ens[i]][0] = np.nansum(gridAreaPeat.where(
                                    talikAnnFEEDBACK[ens[i]] == 0),axis=(0,1))
        for iyear in range(1,55):
            talikAreaEnsCONTROL_peat[ens[i]][iyear] = talikAreaEnsCONTROL[ens[i]][iyear-1] + np.nansum(
                gridAreaPeat.where(talikAnnCONTROL[ens[i]] == iyear),axis=(0,1))
            talikAreaEnsFEEDBACK_peat[ens[i]][iyear] = talikAreaEnsFEEDBACK[ens[i]][iyear-1] + np.nansum(
                gridAreaPeat.where(talikAnnFEEDBACK[ens[i]] == iyear),axis=(0,1))
            
    talikAreaCONTROL_peat = make_ensemble_mean_timeseries(talikAreaEnsCONTROL_peat, 10)
    talikAreaFEEDBACK_peat = make_ensemble_mean_timeseries(talikAreaEnsFEEDBACK_peat, 10)
    
    ## figure
    fig, ax = plt.subplots(1,1, figsize=(9,5), dpi=1200)
    for i in range(len(ens)):
        ax.plot(np.linspace(2015,2069,55),talikAreaEnsCONTROL_peat[ens[i]]/1e6,color='xkcd:light red',label='SSP2-4.5',linestyle='--')
        ax.plot(np.linspace(2035,2069,35),talikAreaEnsFEEDBACK_peat[ens[i]][20:]/1e6,color='xkcd:dark sky blue',label='ARISE-SAI-1.5')
    ax.plot(np.linspace(2015,2069,55),talikAreaCONTROL_peat/1e6,linewidth=2,color='xkcd:dark red',linestyle='--')
    ax.plot(np.linspace(2035,2069,35),talikAreaFEEDBACK_peat[20:]/1e6,linewidth=2,color='xkcd:dark blue')
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='xkcd:light red', lw=2, linestyle='--'),
                    Line2D([0], [0], color='xkcd:dark blue', lw=2)]
    plt.legend(custom_lines, ['SSP2-4.5','ARISE-SAI-1.5'], fancybox=True, fontsize=12)
    plt.xlim([2015,2069])
    plt.ylim([0.4e6,4.8e6])
    ax.set_xticks([2015,2025,2035,2045,2055,2065])
    ax.set_xticklabels(['2015','2025','2035','2045','2055','2065'])
    # ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
    plt.ylabel('Area (km$^2$ x 10$^6$)', fontsize=12)
    plt.title('b) Talik area in peatland', fontsize=14, fontweight='bold')
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.savefig('/Users/arielmor/Desktop/SAI/data/ARISE/figures/talik_area_peatland_timeseries.jpg',
                dpi=1200, bbox_inches='tight')
    
    
    #########################################################
    #### FIGURES: Talik: all ens
    #########################################################  
    ''' SSP ensemble members '''
    fig = plt.figure(figsize=(10,4.8))
    cols = 5; rows = 2
    normalize = mcolors.Normalize(vmin=0, vmax=35)
    
    plottingVar = {}
    for numEns in range(len(ens)):
        longitude = lon
        plottingVar[ens[numEns]],lon2 = add_cyclic_point(talikAnnCONTROL_2035[ens[numEns]],coord=longitude)
        
    for i in range(1, cols*rows +1):
        ## add subplot map
        ax = fig.add_subplot(rows, cols, i, projection=ccrs.NorthPolarStereo())
        cf1 = mapsSubplots(ax,plottingVar[ens[i-1]],lat,lon2,21,normalize,i)
        ax.pcolormesh(lon,lat,landMask[41:,:],transform=ccrs.PlateCarree(),cmap=cmapLand)
        
    ## add a subplot for vertical colorbar
    bottom, top = 0.09, 0.95; left, right = 0.1, 0.99
    fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0, wspace=0.09)
    cbar_ax = fig.add_axes([0.15, 0.05, 0.79, 0.038])
    cbar = plt.colorbar(cf1, cax=cbar_ax, ticks=[0,5,10,15,20,25,30,34], fraction=0.03, orientation='horizontal')
    cbar.ax.set_xticklabels(['2035','2040','2045','2050','2055','2060','2065','2069'])
    cbar.set_label('Talik formation year', fontsize=11, **vfont)
    cbar.ax.tick_params(size=0)
    
    ## Save figure
    plt.suptitle('Timing of talik, SSP2-4.5', fontsize=11.5, y=1, x=0.54, **vfont)
    plt.savefig('/Users/arielmor/Desktop/SAI/data/ARISE/figures/talik_formation_year_ensemble_members_CONTROL_from_2035.jpg', 
                dpi=1200, bbox_inches='tight')
    
    
    ''' ARISE ensemble members '''
    fig = plt.figure(figsize=(10,4.8))
    cols = 5; rows = 2
    normalize = mcolors.Normalize(vmin=0, vmax=35)
    
    plottingVar = {}
    for numEns in range(len(ens)):
        plottingVar[ens[numEns]],lon2 = add_cyclic_point(talikAnnFEEDBACK[ens[numEns]],coord=longitude)
        
    for i in range(1, cols*rows +1):
        ## add subplot map
        ax = fig.add_subplot(rows, cols, i, projection=ccrs.NorthPolarStereo())
        cf1 = mapsSubplots(ax,plottingVar[ens[i-1]],lat,lon2,21,normalize,i)
        ax.pcolormesh(lon,lat,landMask[41:,:],transform=ccrs.PlateCarree(),cmap=cmapLand)
        
    ## add a subplot for vertical colorbar
    bottom, top = 0.09, 0.95; left, right = 0.1, 0.99
    fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0, wspace=0.09)
    cbar_ax = fig.add_axes([0.15, 0.05, 0.79, 0.038])
    cbar = plt.colorbar(cf1, cax=cbar_ax, ticks=[0,5,10,15,20,25,30,34], fraction=0.03, orientation='horizontal')
    cbar.ax.set_xticklabels(['2035','2040','2045','2050','2055','2060','2065','2069'])
    cbar.ax.tick_params(size=0)
    
    ## Save figure
    plt.suptitle('Timing of talik, ARISE-SAI-1.5', fontsize=11.5, y=1, x=0.54, **vfont)
    plt.savefig('/Users/arielmor/Desktop/SAI/data/ARISE/figures/talik_formation_year_ensemble_members_FEEDBACK_from_2035.jpg', 
                dpi=1200, bbox_inches='tight')
    
    
    ''' ---------------------------- '''
    #### Talik: ens diff
    ''' ---------------------------- '''
    brbg_cmap,rdbu_cmap,jet,magma,reds,hot,seismic = get_colormap(51)
    from matplotlib import cm
    import matplotlib.colors as mcolors
    bwr = cm.get_cmap('bwr',((31)))
    
    import pickle
    with open('/Users/arielmor/Desktop/SAI/data/ARISE/data/talikAnnFEEDBACK.pkl', 'rb') as fp:
        talikAnnFEEDBACK = pickle.load(fp)
    with open('/Users/arielmor/Desktop/SAI/data/ARISE/data/talikAnnCONTROL.pkl', 'rb') as fp:
        talikAnnCONTROL = pickle.load(fp)
        
    ## Create subplot figure
    #fig = plt.figure(figsize=(9.9,6.8))
    fig = plt.figure(figsize=(8,7), dpi=1500)
    cols = 4; rows = 3
    norm = mcolors.TwoSlopeNorm(vmin=-15, vcenter=0, vmax=15)
    
    
    
    ## Find difference between control and feedback thaw timing
    thawedControlNotFeedback_mems = {}
    thawedFeedbackNotControl_mems = {}
    talikALWAYS = {}
    diff = {}
    plottingVar = {}
    for numEns in range(len(ens)):
        longitude = lon
        
        ## thawed in control but not in feedback
        thawedControlNotFeedback_mems[ens[numEns]] = talikAnnFEEDBACK[ens[numEns]].copy()
        thawedControlNotFeedback_mems[ens[numEns]][
            (np.isnan(talikAnnFEEDBACK[ens[numEns]])) & 
            (~np.isnan(talikAnnCONTROL[ens[numEns]]))] = 100 # nan in feedback mean = didn't thaw in FB
        thawedControlNotFeedback_mems[ens[numEns]][thawedControlNotFeedback_mems[ens[numEns]] < 100.] = np.nan
        
        ## talik in feedback but not in control:
        thawedFeedbackNotControl_mems[ens[numEns]] = talikAnnCONTROL[ens[numEns]].copy()
        thawedFeedbackNotControl_mems[ens[numEns]][
            (np.isnan(talikAnnCONTROL[ens[numEns]])) & 
            (~np.isnan(talikAnnFEEDBACK[ens[numEns]]))] = -100
        thawedFeedbackNotControl_mems[ens[numEns]][thawedFeedbackNotControl_mems[ens[numEns]] > -100.] = np.nan
        
        ## thawed by 2035:
        talikALWAYS[ens[numEns]] = talikAnnCONTROL[ens[numEns]].copy()
        talikALWAYS[ens[numEns]][
            (talikAnnCONTROL[ens[numEns]] >= 20) |
             (talikAnnFEEDBACK[ens[numEns]] >= 20)] = np.nan
        
        
        diff[ens[numEns]] = talikAnnCONTROL[ens[numEns]] - talikAnnFEEDBACK[ens[numEns]]
        
        ## mask out cells that didn't thaw in control or feedback (diff = nan) and always thawed 
        diff[ens[numEns]][(thawedControlNotFeedback_mems[ens[numEns]] == 100) | (
            thawedFeedbackNotControl_mems[ens[numEns]] == -100) | (
                talikALWAYS[ens[numEns]] < 20)] = np.nan
                
        ## cyclic point to cross 180 longitude
        plottingVar[ens[numEns]],lon2 = add_cyclic_point(diff[ens[numEns]],coord=longitude)
        
    cmapLand = c.ListedColormap(['xkcd:gray','none'])
    for i in range(1, 11):
        ## add subplot map
        ax = fig.add_subplot(rows, cols, i, projection=ccrs.NorthPolarStereo())
        ax, cf1 = mapsSubplotsDiff(ax,plottingVar[ens[i-1]],lat,lon,lon2,
                                   thawedControlNotFeedback_mems[ens[i-1]],
                                   thawedFeedbackNotControl_mems[ens[i-1]],
                                   talikALWAYS[ens[i-1]], norm, i, False)
        ax.pcolormesh(lon,lat,landMask[41:,:],transform=ccrs.PlateCarree(),cmap=cmapLand)
    
    ## add ens mean
    ax = fig.add_subplot(rows, cols, 12, projection=ccrs.NorthPolarStereo())
    ax.set_extent([180, -180, 50, 90], crs=ccrs.PlateCarree())
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes); ax.set_facecolor('0.8')
    ax.pcolormesh(lon,lat,landMask[41:,:],transform=ccrs.PlateCarree(),cmap=cmapLand)
    ax.pcolormesh(lon,lat,thawedControlNotFeedback,transform=ccrs.PlateCarree(),
                        cmap=cMapthawedControlNotFeedback)
    ax.pcolormesh(lon,lat,thawedFeedbackNotControl,transform=ccrs.PlateCarree(),
                        cmap=cMapthawedFeedbackNotControl)
    ax.pcolormesh(lon,lat,talikALWAYSmean,transform=ccrs.PlateCarree(),
                        cmap=cMapALWAYSTHAW)
    cf1 = ax.pcolormesh(lon2,lat,plottingVarMean,transform=ccrs.PlateCarree(), 
                  norm=norm, cmap=bwr) # seismic
    vfont = {'fontname':'Verdana'}
    ax.set_title("EM", fontweight='bold',fontsize=8, y=0.99, **vfont)
    ax.coastlines(linewidth=0.4)
        
    ## add a subplot for vertical colorbar
    bottom, top = 0.09, 0.95; left, right = 0.1, 0.99
    fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0, wspace=0.09)
    cbar_ax = fig.add_axes([0.115, 0.04, 0.86, 0.035])
    cbar = plt.colorbar(cf1, cax=cbar_ax, ticks=[-12, 0, 12], fraction=0.03, orientation='horizontal')
    cbar.ax.set_xticklabels(['Talik forms earlier\nin SSP2-4.5', 'Talik forms\nsame year', 
                             'Talik forms earlier\nin ARISE-SAI-1.5'], **vfont)
    cbar.ax.tick_params(size=0)
    
    ## Save figure
    plt.suptitle('Timing of talik, SSP2-4.5 minus ARISE-SAI', fontsize=12, y=1, x=0.54, **vfont)
    plt.savefig('/Users/arielmor/Desktop/SAI/data/ARISE/figures/control_minus_feedback_talik_year_subplots_2069.jpg', 
                dpi=1500, bbox_inches='tight')
    del fig, ax, cf1, cbar, lon2, bottom, top, left, right
    
    
    #### grid areas for all ensemble members
    gridAreaThawEns = {}
    gridAreaSAIEns  = {}
    gridAreaSSPEns  = {}
    gridAreaPeatEns = {}
    for i in range(len(ens)):
        gridAreaThawEns[ens[i]] = np.array(np.nansum((gridArea[41:,:].where(
            ~np.isnan(thawedControlNotFeedback_mems[ens[i]]))/(1000**2)),axis=(0,1)))
        
        gridAreaSAIEns[ens[i]] = np.array(np.nansum((gridArea[41:,:].where(
            ~np.isnan(thawedFeedbackNotControl_mems[ens[i]]))/(1000**2)),axis=(0,1)))
        
        gridAreaSSPEns[ens[i]] = np.array(np.nansum((gridArea[41:,:].where((
            diff[ens[i]] < 0))/(1000**2)),axis=(0,1)))
        
        gridAreaPeatEns[ens[i]] = np.array(np.nansum((gridArea[41:,:].where(
            (~np.isnan(thawedControlNotFeedback_mems[ens[i]])) & (
                peatland[41:,:].values > 0))/(1000**2)),axis=(0,1)))
    
    print("Talik prevented by SAI (ens mean): ", np.round((
        sum(gridAreaThawEns.values())/10)/1e6, decimals=2), "million km2")
    print("Talik caused by SAI (ens mean): ", np.round((
        sum(gridAreaSAIEns.values())/10)/1e6, decimals=2), "million km2")
    print("Talik delayed by SAI (ens mean): ", np.round((
        sum(gridAreaSSPEns.values())/10)/1e6, decimals=2), "million km2")
    xx = sum(gridAreaSSPEns.values())/10
    print("Percent talik prevented by SAI in peat (ens mean): ", (
        (sum(gridAreaPeatEns.values())/10)/((sum(gridAreaThawEns.values())/10)+(sum(gridAreaSSPEns.values())/10)))*100)
    
    
    
    ''' ------------------------------ '''
    #### Probability of talik prevention
    ''' ------------------------------ '''
    ens = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010']
    import pickle
    with open('/Users/arielmor/Desktop/SAI/data/ARISE/data/talikAnnFEEDBACK.pkl', 'rb') as fp:
        talikAnnFEEDBACK = pickle.load(fp)
    with open('/Users/arielmor/Desktop/SAI/data/ARISE/data/talikAnnCONTROL.pkl', 'rb') as fp:
        talikAnnCONTROL = pickle.load(fp)
        
    probTalikPrevention = {}
    for numEns in range(len(ens)):
        # copy the array where talik formed in control but not arise
        probTalikPrevention[ens[numEns]] = thawedControlNotFeedback_mems[ens[numEns]].copy()
        # put a 0 where talik formed in both, 1 where formed in control but not arise
        probTalikPrevention[ens[numEns]] = xr.where(~np.isnan(thawedControlNotFeedback_mems[ens[numEns]]), 
                                                    1, 0)
        # if talik didn't form in either, put nan
        probTalikPrevention[ens[numEns]] = np.where((np.isnan(talikAnnCONTROL[ens[numEns]])&
                                                      np.isnan(talikAnnFEEDBACK[ens[numEns]])), 
                                                      np.nan, probTalikPrevention[ens[numEns]])
        
    def sum_nan_arrays(a,b,c,d,e,f,g,h,i,j):
        m1 = np.isnan(a)
        m2 = np.isnan(b)
        m3 = np.isnan(c)
        m4 = np.isnan(d)
        m5 = np.isnan(e)
        m6 = np.isnan(f)
        m7 = np.isnan(g)
        m8 = np.isnan(h)
        m9 = np.isnan(i)
        m10 = np.isnan(j)
        return np.where(m1&m2&m3&m4&m5&m6&m7&m8&m9&m10, np.nan, 
                        np.where(m1,0,a) + np.where(m2,0,b) + np.where(m3,0,c) +
                        np.where(m4,0,d) + np.where(m5,0,e) + np.where(m6,0,f) +
                        np.where(m7,0,g) + np.where(m8,0,h) + np.where(m9,0,i) +
                        np.where(m10,0,j))
    
   
    
    talikALWAYS = {}
    for numEns in range(len(ens)):
        longitude = lon
        
        ## thawed in control but not in feedback
        thawedControlNotFeedback_mems[ens[numEns]] = talikAnnFEEDBACK[ens[numEns]].copy()
        thawedControlNotFeedback_mems[ens[numEns]][
            (np.isnan(talikAnnFEEDBACK[ens[numEns]])) & 
            (~np.isnan(talikAnnCONTROL[ens[numEns]]))] = 100 # nan in feedback mean = didn't thaw in FB
        thawedControlNotFeedback_mems[ens[numEns]][thawedControlNotFeedback_mems[ens[numEns]] < 100.] = np.nan
        
        ## talik in feedback but not in control:
        thawedFeedbackNotControl_mems[ens[numEns]] = talikAnnCONTROL[ens[numEns]].copy()
        thawedFeedbackNotControl_mems[ens[numEns]][
            (np.isnan(talikAnnCONTROL[ens[numEns]])) & 
            (~np.isnan(talikAnnFEEDBACK[ens[numEns]]))] = -100
        thawedFeedbackNotControl_mems[ens[numEns]][thawedFeedbackNotControl_mems[ens[numEns]] > -100.] = np.nan
        
        ## thawed by 2035:
        talikALWAYS[ens[numEns]] = talikAnnCONTROL[ens[numEns]].copy()
        talikALWAYS[ens[numEns]][
            (talikAnnCONTROL[ens[numEns]] >= 20) |
             (talikAnnFEEDBACK[ens[numEns]] >= 20)] = np.nan
        
        
    sumForProb = np.full((len(lat),len(lon)), np.nan)
    nonNans = np.full((len(lat),len(lon)), np.nan)
    for ilat in range(len(lat)):
        for ilon in range(len(lon)):
            # number of ens that actually had talik 
            nonNans[ilat,ilon] = np.count_nonzero(~np.isnan(np.dstack((
                                                    probTalikPrevention[ens[0]][ilat,ilon],
                                                    probTalikPrevention[ens[1]][ilat,ilon],
                                                    probTalikPrevention[ens[2]][ilat,ilon],
                                                    probTalikPrevention[ens[3]][ilat,ilon],
                                                    probTalikPrevention[ens[4]][ilat,ilon],
                                                    probTalikPrevention[ens[5]][ilat,ilon],
                                                    probTalikPrevention[ens[6]][ilat,ilon],
                                                    probTalikPrevention[ens[7]][ilat,ilon],
                                                    probTalikPrevention[ens[8]][ilat,ilon],
                                                    probTalikPrevention[ens[9]][ilat,ilon]))))
            # prob of preventing talik in members that had talik in control                                                
            sumForProb[ilat,ilon] = (sum_nan_arrays(probTalikPrevention[ens[0]][ilat,ilon],
                                                    probTalikPrevention[ens[1]][ilat,ilon],
                                                    probTalikPrevention[ens[2]][ilat,ilon],
                                                    probTalikPrevention[ens[3]][ilat,ilon],
                                                    probTalikPrevention[ens[4]][ilat,ilon],
                                                    probTalikPrevention[ens[5]][ilat,ilon],
                                                    probTalikPrevention[ens[6]][ilat,ilon],
                                                    probTalikPrevention[ens[7]][ilat,ilon],
                                                    probTalikPrevention[ens[8]][ilat,ilon],
                                                    probTalikPrevention[ens[9]][ilat,ilon]))/nonNans[ilat,ilon]
    
    # find where there are >= 5 members
    # hatching where there are < 5 members
    lessThan5Members = np.where(nonNans < 5, nonNans, np.nan)
    lessThan5Members = np.where(~np.isnan(sumForProb), lessThan5Members, np.nan)
    
    # hatching where all 10 members had talik
    all10Members = np.where(nonNans == 10, nonNans, np.nan)
    all10Members = np.where(~np.isnan(sumForProb), all10Members, np.nan)
    
    from plottingFunctions import make_maps, get_colormap
    from matplotlib import cm
    import cmasher as cmr
    brbg_cmap,rdbu_cmap,jet,magma,reds,hot,seismic = get_colormap(21)
    
    
    
    
    import cartopy.crs as ccrs
    from cartopy.util import add_cyclic_point
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    import matplotlib.ticker as mticker
    import matplotlib.pyplot as plt
    import matplotlib.path as mpath
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    import numpy as np
    import xarray as xr
    hfont = {'fontname':'Verdana'}
    landMask = landmask()
    
    longitude = lon
    inferno = cm.get_cmap('inferno',(25))
    cmapLand = c.ListedColormap(['xkcd:gray','none'])
    none_map = c.ListedColormap(['none'])
    
    #########################################################
    # land mask
    #########################################################
    datadir = '/Users/arielmor/Desktop/SAI/data/ARISE/data'
    figureDir = '/Users/arielmor/Desktop/SAI/data/ARISE/figures/'
    
    #var,longitude = add_cyclic_point(sumForProb,coord=lon2)
    
    ## Create figure
    fig = plt.figure(figsize=(10,8))
    Norm = mcolors.Normalize(vmin=0, vmax=100)   
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
    ax.set_extent([180, -180, 50, 90], crs=ccrs.PlateCarree())
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.set_facecolor('0.8')
    
    ## field to be plotted
    cf1 = ax.pcolormesh(lon,lat,sumForProb*100.,transform=ccrs.PlateCarree(), 
                  norm=Norm, cmap=inferno)
    ax.coastlines(linewidth=0.8)
    
    ## hatches where < 5 members
    hatch1 = ax.pcolor(lon, lat, lessThan5Members, transform=ccrs.PlateCarree(), cmap=none_map,
                    hatch='+++', edgecolor='xkcd:bright blue', lw=0, zorder=2)
    hatch2 = ax.pcolor(lon, lat, all10Members, transform=ccrs.PlateCarree(), cmap=none_map,
                    hatch='---', edgecolor='xkcd:green', lw=0, zorder=3)
    ## add lat/lon grid lines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), 
                      draw_labels=True,
                      linewidth=1, color='C7', 
                  alpha=0.8, linestyle=':',
                  x_inline=False,
                  y_inline=True,
                  rotate_labels=False)
    gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120])
    gl.ylocator = mticker.FixedLocator([60, 70, 80])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.ylabel_style = {'size': 11, 'rotation':20}
    plt.draw()  # Enable the use of `gl._labels`

    for ea in gl.label_artists:
        # return ea of mpl.text type, e.g. Text(135, 30, '30°N') 
        pos = ea.get_position()
        ## put labels over ocean and not over land
        if pos[0] == 150:
            ea.set_position([0, pos[1]])

        ## vertical cbar on right side
    ## Save figure
    plt.savefig(figureDir + 'Fig9_probability_talik_prevention_feedback_divide_by_non_nans.png',
                dpi=1200, bbox_inches='tight')
    
    
    fig = plt.figure(figsize=(10,8)); ax = fig.add_subplot(1, 1, 1); 
    pc = ax.pcolormesh(sumForProb*100., cmap=inferno); 
    cbar = plt.colorbar(pc, ax=ax, fraction=0.06); 
    cbar.ax.tick_params(labelsize=12); 
    cbar.set_label('Probability of talik prevention (%)', fontsize=13, **hfont); 
    plt.savefig(figureDir + 'colorbar.png',
                dpi=1200, bbox_inches='tight')
    del fig
    
    
    return talikAnnCONTROL, talikAnnFEEDBACK, ens, lat, lon