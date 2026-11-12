#! /usr/bin/env python 

"""

GKU August 2026

Module of python code to map pp climatology and Longhurst coastal provinces for Yanna's publication (Figure 1)

"""

import sys

from calendar import month_abbr

import xarray as xr
import numpy as np
import netCDF4 as netCDF
import pylab as pl
from matplotlib.colors import LogNorm,Normalize
from matplotlib import cm
import matplotlib as mpl

import projmap

import cmocean

# Functions

# Obtain data from netCDF file
def open_netcdf(infile, var_name):
    nc = netCDF.Dataset(infile, 'r') # Open file
    var = nc.variables[var_name][:] # Read variable
    nc.close() # Close file
    return var

# Map products on linear scale
def linear_global_map(lons, lats, data, vmin, vmax, cmap, title, colorbar_text, out_file, year_start, year_end):
    """Global pcolor map using linear colormap

    Create a global map using the 'data' array with a colorbar.

    Parameters
    ----------
    lons : array_like
        vector or array with longitudes (-180 -- 180).
    lats : array_like
        vector or array with latitudes.
    data : array_loke
        array to show as a pcolormesh
    vmin : scalar
        lower limit of the colorbar
    vmax : scalar
        upper limit of the colorbar
    cmap : str or `~matplotlib.colors.Colormap`, default: :rc:`image.cmap`
        A Colormap instance or registered colormap name. The colormap
        maps the *C* values to colors.
    title : str
        Text string for the title of the plot
    colorbar_text : str
        Text string used as legend for the colorbar
    """
    pl.close('all')
    pl.figure(1, (10,8), facecolor='none')
    mp = projmap.Map('glob')
    mp.style['oceancolor'] = '0.8'
    mp.pcolor(lons, lats, data, vmin=vmin, vmax=vmax, cmap=cmap, colorbar=True)
    mp._cb.set_label(colorbar_text, fontsize=28)
    mp._cb.ax.tick_params(labelsize=24)    
    mp.nice()
#    pl.text(0.5, 0.75, title, fontsize=28, horizontalalignment='center', transform=pl.gcf().transFigure)
    pl.savefig(out_file %(year_start, year_end), bbox_inches='tight')
    pl.close()

# Map products on log scale
def log_global_map(lons, lats, data, vmin, vmax, cmap, title, colorbar_text, out_file, year_start, year_end):
    """Global pcolor map using LogNormal colormap

    Create a global map using the 'data' array with a colorbar in log space.

    Parameters
    ----------
    lons : array_like
        vector or array with longitudes (-180 -- 180).
    lats : array_like
        vector or array with latitudes.
    data : array_loke
        array to show as a pcolormesh
    vmin : scalar
        lower limit of the colorbar
    vmax : scalar
        upper limit of the colorbar
    cmap : str or `~matplotlib.colors.Colormap`, default: :rc:`image.cmap`
        A Colormap instance or registered colormap name. The colormap
        maps the *C* values to colors.
    title : str
        Text string for the title of the plot
    colorbar_text : str
        Text string used as legend for the colorbar
    """
    pl.close('all')
    pl.figure(1, (10,8), facecolor='none')
    mp = projmap.Map('glob')
    mp.pcolor(lons, lats, data, colorbar=True,
              norm=LogNorm(vmin=vmin, vmax=vmax), cmap=cmap)
    mp.style['oceancolor'] = '0.8'
    mp._cb.set_label(colorbar_text, fontsize=28)
    mp._cb.ax.tick_params(labelsize=24)    
    mp.nice()
#    pl.text(0.5, 0.75, title, fontsize=28, horizontalalignment='center', transform=pl.gcf().transFigure)
    pl.savefig(out_file %(year_start, year_end), bbox_inches='tight')
    pl.close()


# Create and map climatologies

# Script settings
run_plotting = True
plotting = 'log' # 'log' or 'lin'

# Define variables
years = [1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022] #
months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
coastal_prov = [11,12,13,14,15,16,17,18,19,20,21,22,30,31,32,33,34,35,36,37,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72]

# Define variables and plot characteristics for each variable
variables = ['pp'] #  
var_dict = {'pp':        {'standard_name':'climatology_of_gross_primary_production_of_carbon_in_the_euphotic_zone',
                          'long_name': 'Climatology of primary production in the mixed layer',
                          'units': 'mg C m^-2 d^-1',                          
                          'cmap':'viridis', 
                          'vmin':100.0, 
                          'vmax':5000.0,  
                          'label':'PP (mg C m$^{-2}$ d$^{-1}$)',
                          #'in_file': '/data/datasets/Projects/SCOPE/no-backup/PP_model/9km_output/2023-05-15-CCI_v6_monthly_shubha/filled/pp_processing_%d%02d01_global_filled.nc',
                          'in_file': '/model_out/pp_processing_%d%02d01_global.nc',
                          'out_file_nc': '/.nc',
                          'out_file_png':'.png'}
                          }

# Obtain Longhurst province map and coastal mask
provinces = netCDF.Dataset('.nc')
coastal_mask = provinces.variables['coastal_mask'][:]
coastal_mask = np.flipud(coastal_mask)
#provs = provinces.variables['Region'][:]
#prov_nums = np.unique(provs)

# Loop through variables
for var in variables:
    print(var)
    
    # Map climatologies
    if run_plotting == True:
    
        # Obtain data
        nc = netCDF.Dataset(var_dict[var]['out_file_nc'] %(years[0], years[-1]))
        c_var_mean = nc.variables[var][:]
        lat = nc.variables['lat'][:]
        lon = nc.variables['lon'][:]
        nc.close()
        print(np.min(c_var_mean), np.mean(c_var_mean), np.max(c_var_mean))
        
        # Apply province mask to data
        c_var_mean = np.ma.masked_where(coastal_mask == 0.0, c_var_mean)
        
        # Regrid lat, lon for Robinson projection
        lon, lat = np.meshgrid(lon, lat) 

        # Map climatology
        if plotting == 'log':
            log_global_map(lon, lat, c_var_mean, var_dict[var]['vmin'], var_dict[var]['vmax'], var_dict[var]['cmap'], (str(years[0])+'-'+str(years[-1])), var_dict[var]['label'], var_dict[var]['out_file_png'], years[0], years[-1])
        else:
            linear_global_map(lon, lat, c_var_mean, var_dict[var]['vmin'], var_dict[var]['vmax'], var_dict[var]['cmap'], (str(years[0])+'-'+str(years[-1])), var_dict[var]['label'], var_dict[var]['out_file_png'], years[0], years[-1])
