#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: arielmor

Code for "Natural variability can mask forced permafrost response 
        to stratospheric aerosol injection in the ARISE-SAI-1.5 simulations"
        by A.L. Morrison, E.A. Barnes, and J.W. Hurrell

Publicly available data used in this study are at 
        https://doi.org/10.5065/9kcn-9y79 (ARISE-SAI-1.5) and 
        https://doi.org/10.26024/0cs0-ev98 (SSP2-4.5)
"""

def get_colormap(levs):
    from matplotlib import cm
    from matplotlib.colors import ListedColormap
    import numpy as np
    #########################################################
    # create discrete colormaps from existing continuous maps
    # first make default discrete blue-red colormap 
    # replace center colors with white at 0
    #########################################################
    ## brown-blue
    brbg = cm.get_cmap('BrBG', (levs+3))
    newcolors = brbg(np.linspace(0, 1, 256))
    newcolors[120:136, :] = np.array([1, 1, 1, 1])
    brbg_cmap = ListedColormap(newcolors)
    ## blue-red
    bwr = cm.get_cmap('RdBu_r', (levs+3))
    newcolors = bwr(np.linspace(0, 1, 256))
    newcolors[122:134, :] = np.array([1, 1, 1, 1])
    rdbu_cmap = ListedColormap(newcolors)
    ## rainbow
    jet = cm.get_cmap('turbo', (levs))
    ## other
    magma = cm.get_cmap('magma', (levs))
    reds = cm.get_cmap('Reds',(levs))
    hot = cm.get_cmap('hot',((levs)))
    seismic = cm.get_cmap('seismic',((levs)))
    seismic = ListedColormap(seismic(np.linspace(0.25, 0.75, 128)))
    return brbg_cmap,rdbu_cmap,jet,magma,reds,hot,seismic


def landmask():
    import xarray as xr
    import numpy as np
    #########################################################
    # land mask
    #########################################################
    datadir = '/Users/arielmor/Desktop/SAI/data/ARISE/data'
    ds = xr.open_dataset(datadir + '/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001.clm2.h0.ALT.203501-206912_NH.nc')
    landmask = ds.landmask
    ds.close()
    
    landMask = landmask.where(np.isnan(landmask))
    landMask = landmask.copy() + 2
    landMask = np.where(~np.isnan(landmask),landMask, 1)
    return landMask


def circleBoundary():
    ## gives polar stereographic maps a circular border
    import numpy as np
    import matplotlib.path as mpath
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    return circle


def make_maps(var1,latitude,longitude,vmins,vmaxs,levs,mycmap,label,title,savetitle,extend,addPatch,datadir,figuredir):
    from plottingFunctions import get_colormap, landmask
    from matplotlib import colors as c
    brbg_cmap,rdbu_cmap,jet,magma,reds,hot,seismic = get_colormap(levs)
    cmapLand = c.ListedColormap(['xkcd:gray','none'])
    
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
    
    #########################################################
    # land mask
    #########################################################
    ds = xr.open_dataset(datadir + '/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001.clm2.h0.ALT.203501-206912_NH.nc')
    lat = ds.lat; lon2 = ds.lon
    ds.close()
    
    
    #########################################################
    # make single North Pole stereographic filled contour map
    #########################################################
    # Add cyclic point
    var,lon = add_cyclic_point(var1,coord=longitude)
    
    ## Create figure
    fig = plt.figure(figsize=(10,8))
    if vmins < 0. and vmaxs > 0.:
        norm = mcolors.TwoSlopeNorm(vmin=vmins, vcenter=0, vmax=vmaxs)
    else:
        norm = mcolors.Normalize(vmin=vmins, vmax=vmaxs)
        
    ## Create North Pole Stereo projection map with circle boundary
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
    ax.set_extent([180, -180, 50, 90], crs=ccrs.PlateCarree())
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.set_facecolor('0.8')
    
    ## field to be plotted
    cf1 = ax.pcolormesh(lon,latitude,var,transform=ccrs.PlateCarree(), 
                  norm=norm, cmap=mycmap)
    
    ax.coastlines(linewidth=0.8)
    ## land mask
    ax.pcolormesh(lon2,lat,landMask,transform=ccrs.PlateCarree(),cmap=cmapLand)
        
    
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
    
    if addPatch:
        ## add boxes around important regions for permafrost carbon
        '''canada peat'''
        ax.add_patch(mpatches.Rectangle(xy=[261,50.5],width=20,height=9,
                                        facecolor='none',edgecolor='k',
                                        linewidth=1.5,
                                        transform=ccrs.PlateCarree()))
        '''siberia yedoma'''
        ax.add_patch(mpatches.Rectangle(xy=[110,65.5],width=60,height=10,
                                        facecolor='none',edgecolor='k',
                                        linewidth=1.5,
                                        transform=ccrs.PlateCarree()))
        '''tibet yedoma'''
        ax.add_patch(mpatches.Rectangle(xy=[120,61],width=16,height=4,
                                        facecolor='none',edgecolor='k',
                                        linewidth=1.5,
                                        transform=ccrs.PlateCarree()))
        '''alaska yedoma'''
        ax.add_patch(mpatches.Rectangle(xy=[193,63],width=24,height=9,
                                        facecolor='none',edgecolor='k',
                                        linewidth=1.5,
                                        transform=ccrs.PlateCarree()))

    if mycmap == magma:
        cbar = plt.colorbar(cf1, ax=ax, ticks=[0,10,20,30,40,50], extend=extend, fraction=0.028)
        cbar.ax.set_yticklabels(['2015','2025','2035','2045','2055','2065'], **hfont)
    elif label == ' ':
        cbar = plt.colorbar(cf1, ax=ax, extend=extend, orientation='horizontal',fraction=0.051, 
                            ticks=[0,31,59,90,120,151,181,212,243,273,304,334])
        cbar.ax.tick_params(labelsize=12)
        cbar.ax.set_xticklabels(['Jan 1','Feb 1','Mar 1','Apr 1','May 1',
                                        'Jun 1','Jul 1','Aug 1','Sep 1','Oct 1',
                                        'Nov 1','Dec 1'])
        cbar.ax.tick_params(rotation=30)
    else:
        # horizontal cbar at bottom
        cbar = plt.colorbar(cf1, ax=ax, extend=extend, orientation='horizontal',fraction=0.049)
        cbar.ax.tick_params(labelsize=14)
    cbar.set_label(str(label), fontsize=14, fontweight='bold')
    plt.title(str(title), fontsize=16, fontweight='bold', **hfont, y=1.07)
    ## Save figure
    plt.savefig(figuredir + str(savetitle) + '.png', dpi=1200, bbox_inches='tight')
    return fig, ax


def mapsSubplotsDiff(axs,var,lat,lon,lon2,thawFBnonan,thawCNnonan,thawAlways,norm,i,canada):
    ## used for showing difference in talik formation timing between all 
    ##      ARISE and SSP ensemble members
    import cartopy.crs as ccrs
    from matplotlib import colors as c
    import numpy as np
    import matplotlib.path as mpath
    from plottingFunctions import get_colormap
    brbg_cmap,rdbu_cmap,jet,magma,reds,hot,seismic = get_colormap(51)
    from matplotlib import cm
    bwr = cm.get_cmap('bwr',((31)))
    
    vfont = {'fontname':'Verdana'}
    cMapthawedControlNotFeedback = c.ListedColormap(['xkcd:aqua blue'])
    cMapthawedFeedbackNotControl = c.ListedColormap(['xkcd:bright yellow'])
    cMapALWAYSTHAW = c.ListedColormap(['k'])
    
    if canada: 
        axs.set_extent([-115, -50, 50, 70], crs=ccrs.PlateCarree())
        
    else:
        theta = np.linspace(0, 2*np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        axs.set_extent([180, -180, 50, 90], crs=ccrs.PlateCarree())
        axs.set_boundary(circle, transform=axs.transAxes)
    
    axs.set_facecolor('0.8')

    ## thawed in control but not feedback
    axs.pcolormesh(lon,lat,thawFBnonan,transform=ccrs.PlateCarree(),
                        cmap=cMapthawedControlNotFeedback)
    ## thawed in feedback but not control
    axs.pcolormesh(lon,lat,thawCNnonan,transform=ccrs.PlateCarree(),
                        cmap=cMapthawedFeedbackNotControl)
    ## always thawed = gray
    axs.pcolormesh(lon,lat,thawAlways,transform=ccrs.PlateCarree(),
                        cmap=cMapALWAYSTHAW)
    
    ## difference in thaw timing
    cf1 = axs.pcolormesh(lon2,lat,var,transform=ccrs.PlateCarree(), 
                  norm=norm, cmap=bwr)
    axs.coastlines(linewidth=0.4)
    axs.set_title("Ens 0" + str(i), fontsize=8, y=0.99, **vfont)
    return axs,cf1


def mapsSubplots(axs,var,lat,lon2,levs,norm,i,cmap):
    ## used for ensemble member maps that are NOT talik formation
    import cartopy.crs as ccrs
    import numpy as np
    import matplotlib.path as mpath
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    
    vfont = {'fontname':'Verdana'}
    axs.set_extent([180, -180, 50, 90], crs=ccrs.PlateCarree())
    axs.set_boundary(circle, transform=axs.transAxes)
    axs.set_facecolor('0.8')
    
    ## thaw timing
    cf1 = axs.pcolormesh(lon2,lat,var,transform=ccrs.PlateCarree(), 
                  norm=norm, cmap=cmap)
    axs.coastlines(linewidth=0.4)
    axs.set_title("Ens 0" + str(i), fontsize=7.5, y=0.99, **vfont)
    return cf1
