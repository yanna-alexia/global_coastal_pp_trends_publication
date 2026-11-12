"""
author: yfi 

This script addresses missing monthly primary production (PP) data, where missing values in the mask80 dataset prevent linear
regression and seasonal decomposition. It fills missing values in both the unmasked (pp) and mask80 (pp_mask80) time series using the 
average of the neighbouring months where possible, or the mean for the same calendar month across other years as a fallback,
 then saves the updated data to a new CSV file.
"""

import pandas as pd
import numpy as np

# Load CSV
csv_path = '/path/to/combined_pp_nomask_mask80_filled.csv'
df = pd.read_csv(csv_path)

# Create date column for sorting and indexing
df['date'] = pd.to_datetime(
    dict(year=df['year'], month=df['month'], day=1)
)
df = df.sort_values(['Acronym', 'date'])

# Function to fill missing values within one province
def fill_missing(prov_df, col_name):

    missing_idx = prov_df[prov_df[col_name].isna()].index

    for idx in missing_idx:
        row = prov_df.loc[idx]
        current_date = row['date']
        month = row['month']

        # Previous and next months
        prev_date = current_date - pd.DateOffset(months=1)
        next_date = current_date + pd.DateOffset(months=1)

        prev_val = prov_df.loc[
            prov_df['date'] == prev_date, col_name
        ].values

        next_val = prov_df.loc[
            prov_df['date'] == next_date, col_name
        ].values

        # Use average of neighbouring months if both are available
        if (
            len(prev_val) > 0 and len(next_val) > 0
            and not (np.isnan(prev_val[0]) or np.isnan(next_val[0]))
        ):
            filled = np.nanmean([prev_val[0], next_val[0]])

        else:
            # Fallback: same calendar month across other years
            same_month_vals = prov_df[
                (prov_df['month'] == month) &
                (~prov_df[col_name].isna())
            ][col_name]

            filled = (
                same_month_vals.mean()
                if len(same_month_vals) > 0 else np.nan
            )

        prov_df.at[idx, col_name] = filled

    return prov_df


# Process every province
province_results = []

for province in sorted(df['Acronym'].dropna().unique()):

    print(f"\nProcessing {province}...")

    # Select province
    prov_df = df[df['Acronym'] == province].copy()
    prov_df = prov_df.sort_values('date')

    # Fill both PP columns
    prov_df = fill_missing(prov_df, 'pp')
    prov_df = fill_missing(prov_df, 'pp_mask80')

    # Store updated province data
    province_results.append(prov_df)

    print(
        f"Completed {province}: "
        f"pp missing = {prov_df['pp'].isna().sum()}, "
        f"pp_mask80 missing = {prov_df['pp_mask80'].isna().sum()}"
    )


# Combine all provinces
df_updated = pd.concat(province_results, ignore_index=True)

# Remove temporary date column
df_updated = df_updated.drop(columns='date')

# Save updated CSV
output_path = '/path/to/combined_pp_nomask_mask80_filled_extra.csv'

df_updated.to_csv(output_path, index=False)

print(f"\nAll provinces processed and saved to:\n{output_path}")
