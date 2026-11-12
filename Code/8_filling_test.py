
import pandas as pd
from pathlib import Path

# FILE PATHS

og_file = Path(
    ".csv"
)

filled_file = Path(
    ".csv"
)

output_file = og_file.parent / "gap_filling_summary_by_province.csv"

# LOAD DATA

og = pd.read_csv(og_file)
filled = pd.read_csv(filled_file)

# Columns used to match the same observations
keys = ["year", "month", "provnum", "Acronym"]

variables = ["pp", "pp_mask80"]

# Ensure values are numeric
for col in variables:
    og[col] = pd.to_numeric(og[col], errors="coerce")
    filled[col] = pd.to_numeric(filled[col], errors="coerce")

# MATCH DATASETS

# Check for duplicate observations
assert not og.duplicated(keys).any(), "Duplicates found in OG!"
assert not filled.duplicated(keys).any(), "Duplicates found in filled!"

# Match rows by keys, not row order
df = og.merge(
    filled,
    on=keys,
    how="outer",
    suffixes=("_og", "_filled"),
    indicator=True,
    validate="one_to_one"
)

# Check that both datasets contain the same observations
if not (df["_merge"] == "both").all():
    print("WARNING: The datasets do not contain exactly the same observations.")
    print(df["_merge"].value_counts())

# Create year-month label
df["year_month"] = (
    df["year"].astype("Int64").astype(str)
    + "-"
    + df["month"].astype("Int64").astype(str).str.zfill(2)
)

# CALCULATE GAP-FILLING SUMMARY

results = []

for acronym in sorted(df["Acronym"].dropna().unique()):

    province = df[df["Acronym"] == acronym]

    for variable in variables:

        og_col = variable + "_og"
        filled_col = variable + "_filled"

        # Missing values in original dataset
        og_missing = province[province[og_col].isna()]

        # Missing values after gap filling
        filled_missing = province[province[filled_col].isna()]

        # Months successfully filled
        successfully_filled = province[
            province[og_col].isna() &
            province[filled_col].notna()
        ]

        # Months that were originally available but are now missing
        newly_missing = province[
            province[og_col].notna() &
            province[filled_col].isna()
        ]

        # Format month lists
        og_months = sorted(og_missing["year_month"].tolist())
        filled_months = sorted(filled_missing["year_month"].tolist())
        successful_months = sorted(
            successfully_filled["year_month"].tolist()
        )

        results.append({
            "Acronym": acronym,
            "Variable": variable,

            "OG_missing_count": len(og_missing),
            "OG_missing_months": ", ".join(og_months),

            "Successfully_filled_count": len(successfully_filled),
            "Successfully_filled_months": ", ".join(successful_months),

            "Filled_missing_count": len(filled_missing),
            "Filled_missing_months": ", ".join(filled_months),

            "Newly_missing_count": len(newly_missing),
        })

# CREATE SUMMARY TABLE

summary = pd.DataFrame(results)

# Display complete table
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 300)

print("\nGAP-FILLING SUMMARY BY PROVINCE\n")
print(summary.to_string(index=False))

# SAVE RESULTS

summary.to_csv(output_file, index=False)

print("\nSummary saved to:")
print(output_file)

