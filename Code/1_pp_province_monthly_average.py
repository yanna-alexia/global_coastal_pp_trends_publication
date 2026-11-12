#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: yfi.

- Calculates monthly mean primary production (PP) for each Longhurst province from 1998 to 2022. 
It reads each monthly NetCDF file, extracts PP values for each province, excludes land/undefined provinces (0, 1 and 99) 
and missing or fill-value pixels, and calculates the mean of the remaining valid pixels.

- Records data coverage and saves the results to a CSV. For each province and month, it records the mean PP, 
the number of valid pixels and the total number of pixels in the province. Months with no valid data are skipped.

"""

import os
import numpy as np
import netCDF4 as netCDF
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap



years = range(1998, 2023)
months = range(1, 13)

# Load new province map (once)

province_map_path = "/path/to/longhurst.nc"
pp_data_dir = "/path/to/primary_production_data"
out_file = "monthly_province_PP.csv"
output_dir = "/path/to/output"


#
with netCDF.Dataset(province_map_path) as prov_nc:
    Province = prov_nc.variables['Region'][:]
    Provs = np.unique(Province)

# 
pp_dict = {
    'year': [],
    'month': [],
    'provnum': [],
    'pp': [],
    'valid_pixel_count': [],
    'total_pixel_count': []
}

# # Main loop
for yr in years:
    print(f"Processing year {yr}")
    for mnth in months:
        date_str = f"{yr}{mnth:02d}"
        
        filename_pp = os.path.join(pp_data_dir, f"pp_processing_{date_str}01_global.nc")
        
        try:
            with netCDF.Dataset(filename_pp) as nc:
       
                pp = np.array(nc.variables['pp'][:])  # shape: (2160, 4320)


                for p in Provs:
                    if p in [0, 1, 99]:
                        continue  # Skip land/undefined/non-coastal provinces

                    Prov_index = Province == p
                    prov_pp_values = pp[Prov_index]

                    pp_fill_value = 9.96921e+36


                    valid_mask = (~np.isnan(prov_pp_values)) & (prov_pp_values != pp_fill_value)
                    valid_values = prov_pp_values[valid_mask].copy()

                    # valid_values = valid_values.copy() # copy above added for alsk empty array fix
 

                    valid_pixel_count = valid_values.size
                    total_pixel_count = Prov_index.sum()

                    print(f"Province {p} - {yr}-{mnth:02d}")
                    print(f"Valid pixel count: {valid_pixel_count} / {total_pixel_count}")
                    print(f"Mean pp (valid): {np.nanmean(valid_values)}")

                    # PLOTTING 
                    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                    fig.suptitle(f'Province {p} - {yr}-{mnth:02d}', fontsize=14)

                    # 1. All data (valid + NaN)
                    map_all = np.full_like(Province, np.nan, dtype=np.float32)
                    map_all[Prov_index] = chlor_a[Prov_index]
                    im0 = axes[0].imshow(map_all, origin='lower', cmap='viridis',
                                         norm=colors.LogNorm(vmin=0.01, vmax=10))
                    axes[0].set_title("All Data")
                    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

                    # 2. Valid data only
                    map_valid = np.full_like(Province, np.nan, dtype=np.float32)
                    valid_rows, valid_cols = np.where(Prov_index & ~np.isnan(chlor_a))
                    map_valid[valid_rows, valid_cols] = chlor_a[valid_rows, valid_cols]
                    im1 = axes[1].imshow(map_valid, origin='lower', cmap='viridis',
                                         norm=colors.LogNorm(vmin=0.01, vmax=10))
                    axes[1].set_title("Valid Pixels")
                    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)


                    # 3. NaN data only (highlight missing pixels in bright blue)
                    chl_nan_map = np.full_like(Province, np.nan, dtype=np.float32)
                    nan_mask = Prov_index & np.isnan(chlor_a)
                    chl_nan_map[nan_mask] = 1.0
                    # bright blue colour map for NaNs
                    #blue_cmap = ListedColormap(['#00BFFF'])  # Bright blue
                    #blue_white_cmap = LinearSegmentedColormap.from_list('blue_white', ['#00BFFF', 'white'])
                    blue_black_cmap = LinearSegmentedColormap.from_list('blue_black', ['#00BFFF', 'black'])
                    im2 = axes[2].imshow(chl_nan_map, origin='lower', cmap=blue_black_cmap, vmin=0, vmax=1)
                    axes[2].set_title("NaN Pixels (Missing Data)")
                    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

                    plt.tight_layout()
                    plt.show()

                    # 
                    if valid_pixel_count > 0:
                        pp_dict['year'].append(yr)
                        pp_dict['month'].append(mnth)
                        pp_dict['provnum'].append(p)
                        pp_dict['pp'].append(np.mean(valid_values))
                        pp_dict['valid_pixel_count'].append(valid_pixel_count)
                        pp_dict['total_pixel_count'].append(total_pixel_count)
                    else:
                        print(f"Skipping province {p} due to no valid data.")

        except (IOError, OSError):
            print(f"File not found or unreadable: {filename_chl}")
        except KeyError:
            print(f"Missing variable 'pp' in: {filename_chl}")

# 
df = pd.DataFrame(pp_dict)
df = df[['year', 'month', 'provnum', 'pp', 'valid_pixel_count', 'total_pixel_count']]
df.to_csv(os.path.join(output_dir, out_file), index=False)

print(df.head())
print("Finished processing and saved results.")
