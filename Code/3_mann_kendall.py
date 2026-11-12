"""
author: yfi

This script applies the Mann–Kendall trend test and Sen’s slope estimator to monthly primary production (PP) 
time series for each Longhurst province, analysing unmasked data and data with an 80 mask separately. It calculates 
trend direction, statistical significance, Kendall’s Tau, Sen’s slope and intercept, then saves the results in 
two separate CSV files for full and masked datasets.

"""


import pandas as pd
import pymannkendall as mk
import os

# Input and Output Paths
input_csv = '/path/to/input/combined_pp_data.csv'
output_dir = '/path/to/output/'

# Output file names
pp_output = os.path.join(output_dir, 'pp_mannkendall_sen.csv')
pp_mask80_output = os.path.join(output_dir, 'pp_mask80_mannkendall_sen.csv')

# Load the CSV
df = pd.read_csv(input_csv)

# Create a time variable to sort properly
df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))

# Group by province
group_cols = ['provnum', 'Acronym']
pp_results = []
pp_mask80_results = []

# Process each group
for (provnum, acronym), group in df.groupby(group_cols):
    group_sorted = group.sort_values('date')
    
    # Get the time series
    pp_series = group_sorted['pp'].values
    pp_mask80_series = group_sorted['pp_mask80'].values
    
    # Perform Mann-Kendall and Sen's slope for pp
    mk_result_pp = mk.original_test(pp_series)
    pp_results.append({
        'provnum': provnum,
        'Acronym': acronym,
        'trend': mk_result_pp.trend,
        'h': mk_result_pp.h,
        'p': mk_result_pp.p,
        'z': mk_result_pp.z,
        'Tau': mk_result_pp.Tau,
        'Sen_slope': mk_result_pp.slope,
        'intercept': mk_result_pp.intercept
    })

    # Perform Mann-Kendall and Sen's slope for pp_mask80
    mk_result_mask = mk.original_test(pp_mask80_series)
    pp_mask80_results.append({
        'provnum': provnum,
        'Acronym': acronym,
        'trend': mk_result_mask.trend,
        'h': mk_result_mask.h,
        'p': mk_result_mask.p,
        'z': mk_result_mask.z,
        'Tau': mk_result_mask.Tau,
        'Sen_slope': mk_result_mask.slope,
        'intercept': mk_result_mask.intercept
    })

# Convert results to DataFrames
pp_df = pd.DataFrame(pp_results)
pp_mask80_df = pd.DataFrame(pp_mask80_results)

# Save results
pp_df.to_csv(pp_output, index=False)
pp_mask80_df.to_csv(pp_mask80_output, index=False)

print(f"Done. Files saved:\n- {pp_output}\n- {pp_mask80_output}")
