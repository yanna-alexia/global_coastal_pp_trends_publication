#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yfi

This script compares monthly primary production (PP) time series calculated with no mask and an 80 mask for each Longhurst province, 
using linear regression to assess trends and statistical significance. It plots the time series, regression trends and valid-pixel 
ratios, and saves individual province plots alongside a summary CSV of trend statistics.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
from santer import linregress  # Make sure 'santer' module is accessible

# Load data
csv_path = '//path/to/input/combined_pp_nomask_mask80.csv'
df = pd.read_csv(csv_path)

# Construct datetime column
df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))

# Output directory
output_dir = '/path/to/output/plots'
os.makedirs(output_dir, exist_ok=True)

# Summary list
summary_results = []

# Loop through each province
for (provnum, acronym), group in df.groupby(['provnum', 'Acronym']):
    group = group.sort_values('date')
    group['date_ordinal'] = group['date'].map(pd.Timestamp.toordinal)

    X = group['date_ordinal']
    y_nomask = group['pp']
    y_mask80 = group['pp_mask80']

    # Regressions
    slope0, intercept0, r0, p0, stderr0 = linregress(X, y_nomask)
    slope80, intercept80, r80, p80, stderr80 = linregress(X, y_mask80)

    trend0 = intercept0 + slope0 * X
    trend80 = intercept80 + slope80 * X

    # Plot
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # Top plot: PP time series
    axs[0].plot(group['date'], y_nomask, label='No Mask', marker='o', color='#5e17eb')
    axs[0].plot(group['date'], trend0, linestyle='--', color='#5e17eb')

    axs[0].plot(group['date'], y_mask80, label='Mask 80', marker='s', color='#489f41')
    axs[0].plot(group['date'], trend80, linestyle='--', color='#489f41')

    axs[0].set_title(f'PP Time Series (Provnum: {provnum}, Acronym: {acronym})')
    axs[0].set_ylabel('Primary Production')
    axs[0].legend(loc='upper right')
    axs[0].grid(True)

    stats_text = (
        f'No Mask: Slope={slope0:.2e}, R²={r0**2:.3f}, p={p0:.4f}\n'
        f'Mask 80: Slope={slope80:.2e}, R²={r80**2:.3f}, p={p80:.4f}'
    )
    axs[0].text(0.02, 0.98, stats_text, transform=axs[0].transAxes,
                verticalalignment='top', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))

    # Bottom plot: valid/total ratios
    ratio_nomask = group['valid_pixel_count'] / group['total_pixel_count']
    ratio_80 = group['valid_pixel_count_mask80'] / group['total_pixel_count_mask80']

    axs[1].plot(group['date'], ratio_nomask, label='No Mask', color='#5e17eb')
    axs[1].plot(group['date'], ratio_80, label='Mask 80', color='#489f41')
    axs[1].set_ylabel('Valid Pixel Ratio')
    axs[1].set_xlabel('Date')
    axs[1].legend()
    axs[1].grid(True)

    # Format x-axis
    all_years = pd.DatetimeIndex(group['date']).year.unique()
    exclude_years = {1997, 2023, 2024}
    tick_years = sorted([y for y in all_years if y not in exclude_years])
    tick_dates = [pd.Timestamp(f'{y}-01-01') for y in tick_years]

    for ax in axs:
        ax.set_xticks(tick_dates)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.setp(axs[1].get_xticklabels(), rotation=45, ha='right')

    # Save figure
    out_path = os.path.join(output_dir, f'{acronym}_{provnum}_pp_trend.png')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f'Saved plot: {out_path}')

    # Store stats
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

# Save summary table
summary_df = pd.DataFrame(summary_results)
summary_csv = os.path.join(output_dir, 'pp_trend_summary.csv')
summary_df.to_csv(summary_csv, index=False)
print(f"All plots and summary saved to: {summary_csv}")
