import pandas as pd

# Path to your dataset
csv_path = '/users/rsg-new/yfi/Documents/FOCUS/June_2025/PP/Datasets/combined_pp_nomask_mask80_filled.csv'

# Load CSV
df = pd.read_csv(csv_path)

# Total NaNs per column
print("\n🔍 Total missing values per column:")
print(df.isna().sum())

# Total blank strings per column (object/string columns only)
blank_counts = (df.select_dtypes(include='object') == '').sum()
print("\n🔍 Total blank ('') strings per column:")
print(blank_counts)

# Check by province: NaNs in pp and pp_mask80
print("\n🗺️ Missing values by province for 'pp' and 'pp_mask80':")
pp_nans = df[df['pp'].isna()].groupby('Acronym').size()
pp_mask80_nans = df[df['pp_mask80'].isna()].groupby('Acronym').size()

# Combine into single table
missing_summary = pd.DataFrame({
    'pp_missing': pp_nans,
    'pp_mask80_missing': pp_mask80_nans
}).fillna(0).astype(int)

print(missing_summary)

# Optional: print row
