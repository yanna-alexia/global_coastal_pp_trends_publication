#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Author: yfi

This script applies piecewise linear regression to the seasonally decomposed trend in monthly primary production (PP) 
data for each Longhurst province. It tests models with one to four breakpoints, selects the model with the highest R², 
calculates the slope and statistical significance of each segment using Santer linear regression, and saves province-level 
plots and model-selection summaries.

"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from piecewise_regression import Fit
from santer import linregress
import os
import io
import sys
import re

# Load dataset
df = pd.read_csv('/.csv')
df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))

output_dir = '/'

summary_lines = []
model_selection_results = []


for (provnum, acronym), group in df.groupby(['provnum', 'Acronym']):
    group = group.sort_values('date').set_index('date').asfreq('MS')

    try:
        # Seasonal decomposition (trend only)
        decomposition = seasonal_decompose(group['pp'], model='additive', period=12, extrapolate_trend='freq')
        trend = decomposition.trend.dropna()

        if len(trend) < 24:  # Need enough data points
            summary_lines.append(f"{acronym} ({provnum}): Not enough trend data for regression.\n")
            continue

        x = np.arange(len(trend))
        y = trend.values

        best_r2 = -np.inf
        best_fit = None
        best_n_breakpoints = None
        best_estimates = None
        best_summary = ""

        # Try 1–4 breakpoints
        for n_bkpts in range(1, 5):
            try:
                pw_fit = Fit(x, y, n_breakpoints=n_bkpts)

                # Capture printed summary output
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()
                pw_fit.summary()
                summary_text = sys.stdout.getvalue()
                sys.stdout = old_stdout

                pw_results = pw_fit.get_results()
                if not pw_results or "estimates" not in pw_results:
                    continue

                # Parse R²
                r2_match = re.search(r"R Squared\s+([0-9.]+)", summary_text)
                if not r2_match:
                    continue
                r2 = float(r2_match.group(1))

                if r2 > best_r2:
                    best_r2 = r2
                    best_fit = pw_fit
                    best_n_breakpoints = n_bkpts
                    best_estimates = pw_results["estimates"]
                    best_summary = summary_text

            except Exception as e:
                summary_lines.append(f"{acronym} (bkpts={n_bkpts}) failed: {e}\n")
                continue

        if not best_estimates:
            summary_lines.append(f"{acronym}: No successful regression.\n")
            continue

        # Store model selection info
        model_selection_results.append({
            "Province": acronym,
            "Best_n_breakpoints": best_n_breakpoints,
            "R_squared": best_r2
        })

        # Breakpoint positions
        breakpoints = []
        for key in best_estimates:
            if "breakpoint" in key:
                est = best_estimates[key]["estimate"]
                breakpoints.append(int(round(est)))
        breakpoints = [0] + sorted(breakpoints) + [len(x)]

        summary_lines.append(f"--- {acronym} (best n_breakpoints = {best_n_breakpoints}, R² = {best_r2:.4f}) ---\n")
        summary_lines.append(best_summary + "\n")
        summary_lines.append(f"Breakpoints at indices: {breakpoints[1:-1]}\n")

        # Plot
        plt.figure(figsize=(12, 6))
        trend_dates = trend.index
        plt.plot(trend_dates, y, label='Trend', color='black', alpha=0.6)
        plt.scatter(trend_dates, y, color='grey', s=20)

        colors = ['#5e17eb', '#489f41', '#ffbd59', '#e75d5d']
        for i in range(len(breakpoints) - 1):
            start, end = breakpoints[i], breakpoints[i + 1]
            x_seg = x[start:end]
            y_seg = y[start:end]

            slope, intercept, t_val, p_val, stderr = linregress(x_seg, y_seg)
            y_pred = slope * x_seg + intercept
            linestyle = '-' if p_val <= 0.05 else '--'

            plt.plot(trend_dates[start:end], y_pred,
                     label=f'Segment {i+1}: slope={slope:.3f}, p={p_val:.3g},  R²={best_r2:.3f}',
                     color=colors[i % len(colors)], linestyle=linestyle)

            summary_lines.append(
                f"Segment {i+1}: slope={slope:.5f}, stderr={stderr:.5f}, p={p_val:.3g} ({'significant' if p_val <= 0.05 else 'not significant'})\n"
            )

        plt.title(f'{acronym} - Piecewise Regression on Trend', fontsize=20)
        plt.xlabel('Year', fontsize=16)
        plt.ylabel('Primary Production', fontsize=16)
        plt.legend()
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.xticks(rotation=45, fontsize=14)

        plot_path = os.path.join(output_dir, f'{acronym}_{provnum}_pp_trend_breakpoints.png')
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved: {plot_path}")
        summary_lines.append("\n")

    except Exception as e:
        summary_lines.append(f"{acronym} ({provnum}) skipped due to error: {e}\n")
        continue

# Save model selection summary CSV
selection_df = pd.DataFrame(model_selection_results)
summary_csv_path = os.path.join(output_dir, 'province_model_selection_summary.csv')
selection_df.to_csv(summary_csv_path, index=False)
print(f"Model selection summary saved: {summary_csv_path}")

# Save text summary
summary_txt_path = os.path.join(output_dir, 'trend_breakpoint_summary.txt')
with open(summary_txt_path, 'w') as f:
    f.writelines(summary_lines)

print(f"Summary saved: {summary_txt_path}")
