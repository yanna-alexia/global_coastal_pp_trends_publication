#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

author: yfi.

This script performs seasonal decomposition of monthly primary production (PP) time series for each Longhurst province, 
comparing unmasked data with data using an 80 mask. It extracts the long-term trend components, applies linear regression 
to assess trend direction and statistical significance, and saves comparison plots and a summary CSV containing the trend statistics and valid-pixel ratios.

"""


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
from santer import linregress
from statsmodels.tsa.seasonal import seasonal_decompose
import sys

# Load data
csv_path = 'path/to/input/combined_pp_data.csv'
df = pd.read_csv(csv_path)

# Construct datetime
df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))

# Output directory
output_dir = '/path/to/output/plots/'
os.makedirs(output_dir, exist_ok=True)

summary_results = []


for (provnum, acronym), group in df.groupby(['provnum', 'Acronym']):
    group = group.sort_values('date')

    # Ensure continuous time index
    group = group.set_index('date').asfreq('MS')
    group['date_ordinal'] = group.index.map(pd.Timestamp.toordinal)

    # print(group['pp'])

    try:

        if group['pp'].isna().any():
            missing_pp = group['pp'].isna()
            print("Missing values in 'pp':", missing_pp)
            print(group[missing_pp])

        # Check where 'pp_mask80' has missing values
        if group['pp_mask80'].isna().any():
            missing_pp_mask80 = group['pp_mask80'].isna()
            print("Missing values in 'pp_mask80':")
            print(group[missing_pp_mask80])

        # # Also check combined missing rows
        # if group[['pp', 'pp_mask80']].isna().any():
        #     missing_any = group[['pp', 'pp_mask80']].isna().any()
        #     print("Rows where either 'pp' or 'pp_mask80' is missing:")
        #     print(group[missing_any])

        # Seasonal decomposition
        decomposition_nomask = seasonal_decompose(group['pp'], model='additive', period=12, extrapolate_trend='freq')
        decomposition_mask80 = seasonal_decompose(group['pp_mask80'], model='additive', period=12, extrapolate_trend='freq')

        trend0 = decomposition_nomask.trend
        trend80 = decomposition_mask80.trend

        # Linear regression on trend component
        valid_idx = trend0.notna() & trend80.notna()
        X_trend = group.loc[valid_idx, 'date_ordinal']

        slope0, intercept0, r0, p0, stderr0 = linregress(X_trend, trend0[valid_idx])
        slope80, intercept80, r80, p80, stderr80 = linregress(X_trend, trend80[valid_idx])

        # Generate Santer trend lines
        trend_dates = group.index[valid_idx]
        trend_line0 = intercept0 + slope0 * trend_dates.map(pd.Timestamp.toordinal)
        trend_line80 = intercept80 + slope80 * trend_dates.map(pd.Timestamp.toordinal)

    except Exception as e:
        print(f"Skipping {provnum} - {acronym}: {e}")
        continue

    # Plotting
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # Top panel: time series and trend
    axs[0].plot(group.index, group['pp'], label='PP No Mask', marker='o', linestyle='-', color='#5e17eb', alpha=0.4)
    axs[0].plot(trend0.index, trend0, label='Trend Component (No Mask)', color='#5e17eb', linewidth=2)
    axs[0].plot(trend_dates, trend_line0, label='Santer Trend (No Mask)', color='#5e17eb', linestyle='--')

    axs[0].plot(group.index, group['pp_mask80'], label='PP Mask 80', marker='s', linestyle='-', color='#489f41', alpha=0.4)
    axs[0].plot(trend80.index, trend80, label='Trend Component (Mask 80)', color='#489f41', linewidth=2)
    axs[0].plot(trend_dates, trend_line80, label='Santer Trend (Mask 80)', color='#489f41', linestyle='--')

    axs[0].set_title(f'PP Seasonal Decomposition and Santer Trends\nProvnum: {provnum}, Acronym: {acronym}', fontsize=20)
    axs[0].set_ylabel('Primary Production', fontsize=16)
    axs[0].legend(loc='upper right')
    axs[0].grid(True)

    stats_text = (
        f'No Mask: slope={slope0:.2e}, R²={r0**2:.3f}, p={p0:.4f}\n'
        f'Mask 80: slope={slope80:.2e}, R²={r80**2:.3f}, p={p80:.4f}'
    )
    axs[0].text(0.02, 0.98, stats_text, transform=axs[0].transAxes,
                va='top', ha='left', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))

    # Bottom panel: valid pixel ratios
    ratio_nomask = group['valid_pixel_count'] / group['total_pixel_count']
    ratio_mask80 = group['valid_pixel_count_mask80'] / group['total_pixel_count_mask80']

    axs[1].plot(group.index, ratio_nomask, label='No Mask', color='#5e17eb')
    axs[1].plot(group.index, ratio_mask80, label='Mask 80', color='#489f41')

    axs[1].set_ylabel('Valid Pixel Ratio', fontsize=16)
    axs[1].set_xlabel('Date', fontsize=16)
    axs[1].legend()
    axs[1].grid(True)

    # Format x-axis
    all_years = group.index.year.unique()
    exclude_years = {1997, 2023, 2024}
    tick_years = sorted([y for y in all_years if y not in exclude_years])
    tick_dates = [pd.Timestamp(f'{y}-01-01') for y in tick_years]

    for ax in axs:
        ax.set_xticks(tick_dates)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(axs[1].get_xticklabels(), rotation=45, ha='right', fontsize=14)

    # Save figure
    out_path = os.path.join(output_dir, f'{acronym}_{provnum}_pp_trend_decomposed.png')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f'Saved plot: {out_path}')

    summary_results.append({
        'provnum': provnum,
        'Acronym': acronym,
        'pp_slope_nomask': slope0,
        'pp_r2_nomask': r0**2,
        'p_nomask': p0,
        'pp_slope_mask80': slope80,
        'pp_r2_mask80': r80**2,
        'p_mask80': p80
    })

# Save summary
summary_df = pd.DataFrame(summary_results)
summary_csv = os.path.join(output_dir, 'pp_trend_summary_decomposed.csv')
summary_df.to_csv(summary_csv, index=False)
print(f"Summary saved to {summary_csv}")
