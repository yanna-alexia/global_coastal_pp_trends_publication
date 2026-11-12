"""
author: yfi

This script performs Fast Fourier Transform (FFT) analysis on monthly primary production (PP) time series 
for each Longhurst province to identify periodic patterns and dominant timescales. It log10-transforms PP, 
optionally removes the monthly seasonal cycle and linear trend, calculates and plots amplitude spectra 
against periods in months, and saves the spectral results to a CSV alongside individual and mean spectrum plots.
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 1. SETTINGS

csv_path = ("/path/to/input/")
output_dir = ("/path/to/ouput/")


output_path = os.path.join(
    output_dir, 'fft_all_provinces_corrected_detrend.png'
)

results_path = os.path.join(
    output_dir, 'fft_all_provinces_spectral_results_new_detrended.csv'
)

# Sampling interval: monthly observations
SAMPLING_INTERVAL = 1  # months

# Remove monthly seasonal cycle before FFT?
REMOVE_SEASONALITY = True

# Remove linear long-term trend before FFT?
DETREND = True

# Minimum number of observations required
MIN_OBSERVATIONS = 24


# 2. LOAD AND PREPARE DATA

df = pd.read_csv(csv_path)

# Create datetime column
df['date'] = pd.to_datetime(
    dict(
        year=df['year'],
        month=df['month'],
        day=1
    )
)

# Check required columns
required_columns = ['date', 'Acronym', 'pp']

for col in required_columns:
    if col not in df.columns:
        raise ValueError(
            f"Required column '{col}' is missing from dataset."
        )

# Sort data
df = df.sort_values(['Acronym', 'date'])


# 3. FFT ANALYSIS

plt.figure(figsize=(14, 7))

all_results = []

for province in sorted(df['Acronym'].unique()):

    print(f"\nProcessing {province}...")

    # Select province and sort chronologically
    df_prov = (
        df[df['Acronym'] == province]
        .sort_values('date')
        .copy()
    )

    # Check for duplicate monthly observations
    if df_prov['date'].duplicated().any():

        print(
            f"WARNING: Duplicate monthly observations "
            f"for {province}. Skipping."
        )
        continue

    # Set date as index
    df_prov = df_prov.set_index('date')

    # Check for continuous monthly observations
    expected_dates = pd.date_range(
        start=df_prov.index.min(),
        end=df_prov.index.max(),
        freq='MS'
    )

    if not df_prov.index.equals(expected_dates):

        print(
            f"WARNING: Missing months or irregular dates "
            f"for {province}. Skipping."
        )
        continue

    # Extract PP values
    pp = df_prov['pp'].astype(float)

    # Check for missing values
    if pp.isna().any():

        print(
            f"WARNING: Missing PP values for {province}. "
            f"Skipping."
        )
        continue

    # Check for non-positive values
    if (pp <= 0).any():

        print(
            f"WARNING: Non-positive PP values for {province}. "
            f"Cannot apply log10 transformation. Skipping."
        )
        continue

    # Check minimum record length
    N = len(pp)

    if N < MIN_OBSERVATIONS:

        print(
            f"WARNING: Insufficient observations for "
            f"{province}. Skipping."
        )
        continue

    # 3a. Log10 transformation

    log_pp = np.log10(pp.values)

    # 3b. Remove seasonal cycle (optional)

    if REMOVE_SEASONALITY:

        months = df_prov.index.month

        # Calculate mean for each calendar month
        monthly_climatology = (
            pd.Series(log_pp, index=df_prov.index)
            .groupby(months)
            .mean()
        )

        # Subtract corresponding monthly climatology
        seasonal_component = np.array([
            monthly_climatology.loc[m]
            for m in months
        ])

        processed_pp = log_pp - seasonal_component

    else:

        processed_pp = log_pp.copy()

    # 3c. Remove linear trend (optional)

    if DETREND:

        time = np.arange(N)

        coefficients = np.polyfit(
            time,
            processed_pp,
            1
        )

        trend = np.polyval(
            coefficients,
            time
        )

        processed_pp = processed_pp - trend

    # 3d. Remove mean

    processed_pp = processed_pp - np.mean(processed_pp)

    # 3e. Calculate FFT

    fft_result = np.fft.rfft(processed_pp)

    # Physical frequencies in cycles per month
    frequencies = np.fft.rfftfreq(
        N,
        d=SAMPLING_INTERVAL
    )

    # 3f. Calculate one-sided amplitude spectrum

    amplitudes = np.abs(fft_result) / N

    # Double positive-frequency amplitudes to account
    # for the omitted negative-frequency half.
    if N % 2 == 0:

        amplitudes[1:-1] *= 2

    else:

        amplitudes[1:] *= 2

    # 3g. Exclude zero frequency

    valid = frequencies > 0

    frequencies = frequencies[valid]
    amplitudes = amplitudes[valid]

    # 3h. Convert frequency to period in months

    periods = 1 / frequencies

    # Sort by increasing period
    order = np.argsort(periods)

    periods = periods[order]
    amplitudes = amplitudes[order]

    # 3i. Plot spectrum

    plt.plot(
        periods,
        amplitudes,
        label=province,
        alpha=0.7,
        linewidth=1
    )

    # 3j. Save spectral results

    province_results = pd.DataFrame({

        'Province': province,

        'Frequency_cycles_per_month': frequencies[order],

        'Period_months': periods,

        'Amplitude': amplitudes

    })

    all_results.append(province_results)

    print(
        f"Completed {province}: "
        f"{N} monthly observations."
    )


# 4. FINALISE AND SAVE FIGURE

plt.title(
    'FFT Amplitude Spectra for All Provinces',
    fontsize=14
)

plt.xlabel(
    'Period (months)',
    fontsize=12
)

plt.ylabel(
    'FFT Amplitude (log10 PP units)',
    fontsize=12
)

plt.grid(
    True,
    alpha=0.4
)

plt.legend(
    loc='upper right',
    fontsize='small',
    ncol=2
)

plt.tight_layout()

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches='tight'
)
plt.show()
#plt.close()

print("\nFFT analysis completed.")

print(f"Figure saved to: {output_path}")


# 5. SAVE ALL SPECTRAL RESULTS

if all_results:

    results_df = pd.concat(
        all_results,
        ignore_index=True
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    print(
        f"Spectral results saved to: {results_path}"
    )

else:

    print(
        "No valid provincial spectra were generated."
    )

## extra plot:


# Combine all provincial spectra
results_df = pd.read_csv(results_path)

# Calculate mean amplitude at each period
mean_spectrum = (
    results_df.groupby('Period_months')['Amplitude']
    .agg(['mean', 'std'])
    .reset_index()
)

# Plot mean spectrum
plt.figure(figsize=(11, 6))

plt.plot(
    mean_spectrum['Period_months'],
    mean_spectrum['mean'],
    color='darkblue',
    linewidth=2,
    label='Mean across provinces'
)

plt.fill_between(
    mean_spectrum['Period_months'],
    mean_spectrum['mean'] - mean_spectrum['std'],
    mean_spectrum['mean'] + mean_spectrum['std'],
    color='steelblue',
    alpha=0.2,
    label='±1 standard deviation'
)

plt.xlabel('Period (months)')
plt.ylabel('Normalised FFT Amplitude')
plt.title('Mean FFT Spectrum Across Provinces')

plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, 'fft_mean_spectrum_new_detrend.png'),
    dpi=300
)

plt.show()


