#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author: yfi

This script performs pixel-by-pixel linear regression on log10-transformed primary production (PP) data within 
each Longhurst province to calculate long-term trends and percentage changes. It generates spatial maps of 
percentage change, regression slope, R², p-values and valid observation counts, and saves the percentage-change 
results as NetCDF files for each province.
"""

import numpy as np
import xarray as xr
import pandas as pd
import os
from santer import linregress
import matplotlib.pyplot as plt

# File paths
province_csv_path = "/path/to/input/"
pp_path = "/path/to/input/"
region_path = "/path/to/input/"
output_base = "/path/to/output/"  
os.makedirs(output_base, exist_ok=True)

# Load datasets
ds_pp = xr.open_dataset(pp_path)
ds_region = xr.open_dataset(region_path)

region = ds_region['Region']
pp_raw = ds_pp['pp']  # correct variable name from NetCDF for pp
pp_raw = pp_raw.where(pp_raw < 1e35, np.nan)  # Mask absurd fill values
time = ds_pp['time'].values
time_numeric = np.arange(len(time))

# Define coordinates
lat_len, lon_len = region.shape
lat = np.linspace(90, -90, lat_len)[::-1]
lon = np.linspace(0, 360, lon_len, endpoint=False)
lon = ((lon + 180) % 360) - 180
sort_idx = np.argsort(lon)
lon = lon[sort_idx]
coords = {'lat': lat, 'lon': lon}

# Roll arrays to centre longitude at 0°
zero_idx = np.where(lon == 0)[0][0]
roll_amount = (region.shape[1] // 2) - zero_idx
region_rolled = np.roll(region.values, shift=roll_amount, axis=1)
pp_rolled = np.roll(pp_raw.values, shift=roll_amount, axis=2)

# Plotting function
def plot_map_save(data, title, filename, out_dir, cmap='viridis', vmin=None, vmax=None,
                  lat_min=-90, lat_max=90, lon_min=-180, lon_max=180):
    
    lat_mask = (lat >= lat_min) & (lat <= lat_max)
    lon_mask = (lon >= lon_min) & (lon <= lon_max)
    lat_sel = lat[lat_mask]
    lon_sel = lon[lon_mask]
    data_sel = data[np.ix_(lat_mask, lon_mask)]

    plt.figure(figsize=(12, 5))
    extent = [lon_sel.min(), lon_sel.max(), lat_sel.min(), lat_sel.max()]
    plt.imshow(data_sel, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)
    plt.colorbar(label=title)
    plt.title(title)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=150)
    plt.close()

# Load province metadata
df = pd.read_csv(province_csv_path)

# List of provinces to process
#provinces_to_run = ['GUIN', 'INDE', 'INDWE', 'NECS', 'NEWZ', 'NWCS', 'REDS', 'SUND']  # Acronyms of the provinces you want to process

# Filter dataframe to select only these provinces
#df_selected = df[df['Acronym'].isin(provinces_to_run)]

# Main loop over provinces
for idx, row in df.iterrows():
    province_name = row['Province']
    province_id = row['Number']
    lat_min = row['lat_min']
    lat_max = row['lat_max']
    lon_min = row['lon_min']
    lon_max = row['lon_max']

    # Skip if bounding box values are missing
    if pd.isnull(lat_min) or pd.isnull(lat_max) or pd.isnull(lon_min) or pd.isnull(lon_max):
        print(f"Skipping province {province_name} (missing lat/lon bounds).")
        continue

    print(f"Processing province: {province_name} (ID: {province_id})")

    mask = (region_rolled == province_id)
    pp_masked = np.where(mask, pp_rolled, np.nan)

    slope = np.full((lat_len, lon_len), np.nan)
    intercept = np.full((lat_len, lon_len), np.nan)
    r_value = np.full((lat_len, lon_len), np.nan)
    p_value = np.full((lat_len, lon_len), np.nan)
    n_valid = np.zeros((lat_len, lon_len), dtype=int)


# Perform regression per grid cell log10
    r2 = r_value ** 2
    mean_val = np.nanmean(pp_masked, axis=0)
    num_years = len(time)
    pct_change = np.full((lat_len, lon_len), np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        pct_change = (slope * num_years / mean_val) * 100
    pct_change[~mask] = np.nan

    for i in range(lat_len):
        for j in range(lon_len):
            if mask[i, j]:
                y = pp_masked[:, i, j]
                valid = ~np.isnan(y) & (y > 0)  # log10 undefined for 0 or negative
                n = np.sum(valid)
                n_valid[i, j] = n
                if n > 10:
                    x = time_numeric[valid]
                    y_valid = np.log10(y[valid])  # Apply log10 transformation
                    try:
                        result = linregress(x, y_valid)
                        slope[i, j], intercept[i, j], r_value[i, j], p_value[i, j] = result[:4]
                    except:
                        pass

    r2 = r_value ** 2
    num_years = len(time)

    # Calculate % change in log-space: (10^(slope * num_years) - 1) * 100
    pct_change = np.full((lat_len, lon_len), np.nan)
    with np.errstate(over='ignore', invalid='ignore'):
        pct_change = (10 ** (slope * num_years) - 1) * 100

    pct_change[~mask] = np.nan

    print("Non-NaN slope count:", np.count_nonzero(~np.isnan(slope)))
    print("Non-NaN mean_val count:", np.count_nonzero(~np.isnan(mean_val)))
    print("Non-NaN pct_change count:", np.count_nonzero(~np.isnan(pct_change)))

    # Output directory for this province
    province_dir = os.path.join(output_base, row['Acronym'])
    os.makedirs(province_dir, exist_ok=True)

    # Save NetCDF file
    pct_da = xr.DataArray(pct_change, dims=('lat', 'lon'), coords=coords)
    pct_da.to_netcdf(os.path.join(province_dir, f"{row['Acronym']}_PP_pct_change.nc"))

    # Save plots
    plot_map_save(pct_change, 'Total % Change (pp)', 'pp_pct_change.png', province_dir,
                  cmap='bwr', vmin=-1, vmax=1,
                  lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)

    plot_map_save(slope, 'Regression Slope', 'slope.png', province_dir,
                  lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)

    plot_map_save(r2, 'R²', 'r_squared.png', province_dir,
                  cmap='viridis', vmin=0, vmax=1,
                  lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)

    plot_map_save(p_value, 'p-value', 'p_value.png', province_dir,
                  cmap='inferno', vmin=0, vmax=0.05,
                  lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)

    plot_map_save(n_valid, 'Valid Points', 'valid_pixel_count.png', province_dir,
                  cmap='plasma', lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)
