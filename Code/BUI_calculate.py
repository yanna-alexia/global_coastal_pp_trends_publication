
"""
Author: yfi


This script calculates the Bakun Upwelling Index (BUI) using monthly Ekman transport data for selected or all Longhurst provinces,
 combining coastal point locations, coastline angles and Ekman transport components to estimate offshore transport. It handles 
 missing and invalid Ekman data by finding the nearest grid cell with valid paired transport components, records the distance 
 to that cell for quality control, and saves both coastal-point-level BUI values and province-level monthly averages with 
 valid and missing observation counts.
"""

import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime
import argparse
import os

from scipy.spatial import cKDTree

# PARSE ARGUMENTS
parser = argparse.ArgumentParser(description="Calculate BUI for a specific Longhurst province or all provinces.")
parser.add_argument('--province', type=int, default=None, help="Province number to run BUI calculation for.")
args = parser.parse_args()
target_province = args.province

#  INPUT PATHS 
angles_csv = "/path/to/input/"
longhurst_nc = "/path/to/input/"
ekman_transport_file = "/path/to/input/"

#  OUTPUT PATHS 
base_output_dir = "/path/to/output/"

# Always write to the same files to aggregate all provinces
#output_csv = os.path.join(base_output_dir, "allpointslist_bui_allmonths_ekman_14.csv")
#avg_output_csv = os.path.join(base_output_dir, "province_mean_BUI_ALLMONTHS_ekman_14.csv")

##new
output_csv = os.path.join(base_output_dir, "allpointslist_bui_allmonths_ekman_QC.csv")
avg_output_csv = os.path.join(base_output_dir, "province_mean_BUI_ALLMONTHS_ekman_QC.csv")

def upwell(ektrx, ektry, coast_angle):
    pi = 3.1415927
    degtorad = pi / 180.
    alpha = (360 - coast_angle) * degtorad
    s1 = np.cos(alpha)
    t1 = np.sin(alpha)
    perp = (s1 * ektrx) + (t1 * ektry)
    return perp / 10

def nearest_valid_ekman(ektrx, ektry, coast_lat, coast_lon):
    """
    Find the nearest grid cell with valid values for BOTH
    Ekman transport components for each coastal point.

    Returns:
        ektrx_nearest: nearest valid ektrx values
        ektry_nearest: nearest valid ektry values
        distance_km: distance to the selected grid cell
    """

    # Extract grid data
    x = ektrx.values
    y = ektry.values

    # Only use cells where BOTH components are valid
    valid = np.isfinite(x) & np.isfinite(y)

    if not np.any(valid):
        n = len(coast_lat)
        return (
            np.full(n, np.nan),
            np.full(n, np.nan),
            np.full(n, np.nan)
        )

    # Coordinates of valid grid cells
    lat_grid, lon_grid = np.meshgrid(
        ektrx.latitude.values,
        ektrx.longitude.values,
        indexing="ij"
    )

    valid_lats = lat_grid[valid]
    valid_lons = lon_grid[valid]

    # Convert latitude/longitude to 3D unit-sphere coordinates
    def to_xyz(lat, lon):
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)

        return np.column_stack([
            np.cos(lat_rad) * np.cos(lon_rad),
            np.cos(lat_rad) * np.sin(lon_rad),
            np.sin(lat_rad)
        ])

    valid_xyz = to_xyz(valid_lats, valid_lons)
    coast_xyz = to_xyz(coast_lat, coast_lon)

    # Find nearest valid grid cell
    tree = cKDTree(valid_xyz)
    chord_dist, nearest_idx = tree.query(coast_xyz, k=1)

    # Retrieve the paired Ekman values
    valid_x = x[valid]
    valid_y = y[valid]

    ektrx_nearest = valid_x[nearest_idx]
    ektry_nearest = valid_y[nearest_idx]

    # Convert spherical chord distance to kilometres
    earth_radius_km = 6371.0
    distance_km = (
        2 * earth_radius_km *
        np.arcsin(np.clip(chord_dist / 2, 0, 1))
    )

    return ektrx_nearest, ektry_nearest, distance_km

#   1: Load Coastal Points 
coast_df = pd.read_csv(angles_csv)
coast_df = coast_df.dropna(subset=["angle", "lat", "lon"])

#   2: Load Longhurst Province Grid and Assign to Points 
lh_ds = xr.open_dataset(longhurst_nc)
province_grid = lh_ds['Region']

coast_df["province"] = province_grid.interp(
    lat=("points", coast_df["lat"].values),
    lon=("points", coast_df["lon"].values),
    method="nearest"
).values

coast_df = coast_df.dropna(subset=["province"])
coast_df["province"] = coast_df["province"].astype(int)

#   3: Load Ekman Transport Dataset 
#ek_ds = xr.open_dataset(ekman_transport_file)

 ##new
# 3: Load Ekman Transport Dataset
ek_ds = xr.open_dataset(ekman_transport_file)

# Documented missing-data latitude rows
known_gap_lats = [
    -89.5, -88.5, -10.5, 10.5, 88.5, 89.5
]

# Mask the documented gaps in both Ekman components
for var in ["ektrx", "ektry"]:

    # Count extreme values before masking
    extreme_count = int(
        (np.isfinite(ek_ds[var]) &
         (np.abs(ek_ds[var]) >= 1e10)).sum().values
    )

    # Mask documented missing-data rows
    ek_ds[var] = ek_ds[var].where(
        ~ek_ds.latitude.isin(known_gap_lats)
    )

    # Additional safeguard against extreme invalid values
    ek_ds[var] = ek_ds[var].where(
        np.isfinite(ek_ds[var]) &
        (np.abs(ek_ds[var]) < 1e10)
    )

    print(
        f"{var}: masked documented latitude gaps; "
        f"{extreme_count} extreme values detected."
    )

    # DIAGNOSTIC: Check how much valid data remains after masking ##
    print(f"\nChecking {var} after masking:")

    for date in ["2015-07", "2016-05", "2017-01"]:

        test_data = ek_ds[var].sel(time=date).values

        print(
            date,
            "Finite values:", np.isfinite(test_data).sum(),
            "NaN values:", np.isnan(test_data).sum(),
            "Minimum:", np.nanmin(test_data),
            "Maximum:", np.nanmax(test_data)
        )

    print("\nActual latitude coordinates:")
    print(ek_ds.latitude.values)

    print("\nDocumented gap coordinates:")
    print(known_gap_lats)##



# Fix longitude from 0–360 to -180–180
ek_ds = ek_ds.assign_coords(
    longitude=(((ek_ds.longitude + 180) % 360) - 180)
).sortby('longitude')

ek_times = pd.to_datetime(ek_ds['time'].values)

#   4: Loop Over Time and Provinces 
all_results = []

province_list = [target_province] if target_province is not None else coast_df["province"].unique()
##
print("Target province:", target_province)
print("Provinces being processed:", province_list) ##

for t_idx, t_val in enumerate(ek_times):
    print(f"\nProcessing month: {t_val.strftime('%Y-%m')} ({t_idx + 1}/{len(ek_times)})")

    ektrx = ek_ds['ektrx'].isel(time=t_idx)
    ektry = ek_ds['ektry'].isel(time=t_idx)

    ##
        # DIAGNOSTIC: Check raw Ekman data for the first month only
    if t_idx == 0:
        print("\n== RAW EKMAN DATA DIAGNOSTIC ===")

        print("Month:", t_val.strftime("%Y-%m"))

        print(
            "Finite raw ektrx grid cells:",
            np.isfinite(ektrx.values).sum()
        )

        print(
            "Finite raw ektry grid cells:",
            np.isfinite(ektry.values).sum()
        )

        print(
            "Raw ektrx minimum:",
            np.nanmin(ektrx.values)
        )

        print(
            "Raw ektrx maximum:",
            np.nanmax(ektrx.values)
        )

        print(
            "Raw ektry minimum:",
            np.nanmin(ektry.values)
        )

        print(
            "Raw ektry maximum:",
            np.nanmax(ektry.values)
        )

        print("===========\n")

        ##

    for province in province_list: #
        sub_df = coast_df[coast_df["province"] == province].reset_index(drop=True)

                # DIAGNOSTIC: Check whether nearest grid cells contain valid data
        if t_idx == 0:

            nearest_x = ektrx.sel(
                latitude=xr.DataArray(
                    sub_df["lat"].values, dims="points"
                ),
                longitude=xr.DataArray(
                    sub_df["lon"].values, dims="points"
                ),
                method="nearest"
            )

            nearest_y = ektry.sel(
                latitude=xr.DataArray(
                    sub_df["lat"].values, dims="points"
                ),
                longitude=xr.DataArray(
                    sub_df["lon"].values, dims="points"
                ),
                method="nearest"
            )

            print("\n===== NEAREST GRID VALIDITY =====")

            print(
                "Total coastal points:",
                len(sub_df)
            )

            print(
                "Valid nearest ektrx:",
                np.isfinite(nearest_x.values).sum()
            )

            print(
                "Valid nearest ektry:",
                np.isfinite(nearest_y.values).sum()
            )

            print("\nFirst 20 nearest ektrx values:")
            print(nearest_x.values[:20])

            print("\nFirst 20 nearest ektry values:")
            print(nearest_y.values[:20])

            print("===============\n")

                        # Inspect a 3 x 3 grid around the first coastal point
            test_lat = sub_df["lat"].iloc[0]
            test_lon = sub_df["lon"].iloc[0]

            print("\n===== 3 x 3 GRID NEIGHBOURHOOD =====")

            print("ektrx:")
            print(
                ektrx.sel(
                    latitude=slice(-1.5, 2.5),
                    longitude=slice(-51.5, -47.5)
                ).values
            )

            print("ektry:")
            print(
                ektry.sel(
                    latitude=slice(-1.5, 2.5),
                    longitude=slice(-51.5, -47.5)
                ).values
            )

            print("====================================\n")

        ##
                # DIAGNOSTIC: Check nearest Ekman grid cell for first coastal point
        if t_idx == 0:

            test_lat = sub_df["lat"].iloc[0]
            test_lon = sub_df["lon"].iloc[0]

            print("\n===== NEAREST GRID CELL DIAGNOSTIC =====")

            print("Coastal point latitude:", test_lat)
            print("Coastal point longitude:", test_lon)

            print(
                "Nearest Ekman latitude:",
                ektrx.sel(latitude=test_lat, method="nearest").latitude.values
            )

            print(
                "Nearest Ekman longitude:",
                ektrx.sel(longitude=test_lon, method="nearest").longitude.values
            )

            print(
                "Nearest raw ektrx value:",
                ektrx.sel(
                    latitude=test_lat,
                    longitude=test_lon,
                    method="nearest"
                ).values
            )

            print(
                "Nearest raw ektry value:",
                ektry.sel(
                    latitude=test_lat,
                    longitude=test_lon,
                    method="nearest"
                ).values
            )

            print("========\n")

                        # DIAGNOSTIC: Find nearest valid paired Ekman grid cells
            print("\n===== SEARCH FOR NEAREST VALID EKMAN CELLS =====")

            # Get the raw grid for the current month
            x_grid = ektrx.values
            y_grid = ektry.values

            # A cell is valid only when both components are finite
            valid_mask = np.isfinite(x_grid) & np.isfinite(y_grid)

            lat_grid, lon_grid = np.meshgrid(
                ektrx.latitude.values,
                ektrx.longitude.values,
                indexing="ij"
            )

            valid_lats = lat_grid[valid_mask]
            valid_lons = lon_grid[valid_mask]

            valid_x = x_grid[valid_mask]
            valid_y = y_grid[valid_mask]

            # Distance from the first coastal point
            test_lat = sub_df["lat"].iloc[0]
            test_lon = sub_df["lon"].iloc[0]

            lat1 = np.radians(test_lat)
            lat2 = np.radians(valid_lats)
            dlat = lat2 - lat1
            dlon = np.radians(valid_lons - test_lon)

            a = (
                np.sin(dlat / 2)**2
                + np.cos(lat1) * np.cos(lat2)
                * np.sin(dlon / 2)**2
            )

            distance_km = (
                6371 * 2 * np.arcsin(np.sqrt(a))
            )

            nearest_indices = np.argsort(distance_km)[:10]

            print("First coastal point:", test_lat, test_lon)
            print("Number of valid paired grid cells:", len(valid_lats))

            print("\n10 nearest valid grid cells:")

            for idx in nearest_indices:
                print(
                    f"Latitude: {valid_lats[idx]:.1f}, "
                    f"Longitude: {valid_lons[idx]:.1f}, "
                    f"Distance: {distance_km[idx]:.1f} km, "
                    f"ektrx: {valid_x[idx]:.2f}, "
                    f"ektry: {valid_y[idx]:.2f}"
                )

            for radius in [100, 250, 500, 1000]:
                print(
                    f"Valid cells within {radius} km:",
                    np.sum(distance_km <= radius)
                )

            print("================================================\n")

            ##

        print("\n===== COORDINATE DIAGNOSTIC =====")

        print("Province:", province)

        print("\nCoastal point latitude range:")
        print(sub_df["lat"].min(), "to", sub_df["lat"].max())

        print("\nCoastal point longitude range:")
        print(sub_df["lon"].min(), "to", sub_df["lon"].max())

        print("\nFirst 10 coastal coordinates:")
        print(sub_df[["lat", "lon"]].head(10))

        print("\nEkman latitude range:")
        print(ek_ds.latitude.min().values, "to", ek_ds.latitude.max().values)

        print("\nEkman longitude range:")
        print(ek_ds.longitude.min().values, "to", ek_ds.longitude.max().values)

        print("\nCoastal points outside Ekman longitude range:")
        print(
            (
                (sub_df["lon"] < ek_ds.longitude.min().values) |
                (sub_df["lon"] > ek_ds.longitude.max().values)
            ).sum()
        )

        print("\nCoastal points outside Ekman latitude range:")
        print(
            (
                (sub_df["lat"] < ek_ds.latitude.min().values) |
                (sub_df["lat"] > ek_ds.latitude.max().values)
            ).sum()
        )

        print("==========\n")

        ##

        if len(sub_df) == 0: #
            print(f"Skipping province {province}: No coastal points found.") #
            continue

        # interpolated_ektrx = ektrx.interp(
        #     latitude=("points", sub_df["lat"].values),
        #     longitude=("points", sub_df["lon"].values),
        #     method='nearest'
        # )
        # interpolated_ektry = ektry.interp(
        #     latitude=("points", sub_df["lat"].values),
        #     longitude=("points", sub_df["lon"].values),
        #     method='nearest'
        # )

        # Find nearest grid cell with valid paired Ekman values
        ektrx_values, ektry_values, distance_km = nearest_valid_ekman(
            ektrx,
            ektry,
            sub_df["lat"].values,
            sub_df["lon"].values
        )

        # Calculate BUI using nearest valid Ekman values
        bui_values = upwell(
            ektrx_values,
            ektry_values,
            sub_df["angle"].values
        )

        # Distance diagnostic for first month
        if t_idx == 0:
            print("\n===== NEAREST VALID EKMAN DISTANCE =====")
            print("Minimum distance (km):", np.nanmin(distance_km))
            print("Mean distance (km):", np.nanmean(distance_km))
            print("Maximum distance (km):", np.nanmax(distance_km))
            print("===============\n")

        # DIAGNOSTIC: Check nearest valid Ekman values for first month only
        if t_idx == 0:

            print("\n===== NEAREST VALID EKMAN DIAGNOSTIC =====")

            print("Province:", province)
            print("Number of coastal points:", len(sub_df))

            print(
                "Valid nearest ektrx:",
                np.isfinite(ektrx_values).sum()
            )

            print(
                "Valid nearest ektry:",
                np.isfinite(ektry_values).sum()
            )

            print(
                "Valid BUI values:",
                np.isfinite(bui_values).sum()
            )

            print("\nExample nearest ektrx values:")
            print(ektrx_values[:10])

            print("\nExample nearest ektry values:")
            print(ektry_values[:10])

            print("\nExample BUI values:")
            print(bui_values[:10])

            print("\nExample distances (km):")
            print(distance_km[:10])

            print("===============================\n")

        # Calculate upwelling for all coastal points
        # for i in range(len(sub_df)):
        #     bui = upwell(
        #         interpolated_ektrx.values[i],
        #         interpolated_ektry.values[i],
        #         sub_df["angle"].iloc[i]
        #     )
        #     all_results.append({
        #         "id": sub_df["id"].iloc[i],
        #         "lat": sub_df["lat"].iloc[i],
        #         "lon": sub_df["lon"].iloc[i],
        #         "province": province,
        #         "time": t_val,
        #         "upwelling_index_ekman": bui
        #     })

        # ##new
        # # Calculate upwelling for all coastal points
        # for i in range(len(sub_df)):

        #     x = interpolated_ektrx.values[i]
        #     y = interpolated_ektry.values[i]

        #     # Only calculate BUI when both components are valid
        #     if np.isfinite(x) and np.isfinite(y):

        #         bui = upwell(
        #             x,
        #             y,
        #             sub_df["angle"].iloc[i]
        #         )

        #         if not np.isfinite(bui):
        #             bui = np.nan

        #     else:
        #         bui = np.nan

        #     all_results.append({
        #         "id": sub_df["id"].iloc[i],
        #         "lat": sub_df["lat"].iloc[i],
        #         "lon": sub_df["lon"].iloc[i],
        #         "province": province,
        #         "time": t_val,
        #         "upwelling_index_ekman": bui
        #     })

        # Save BUI results for all coastal points
        for i in range(len(sub_df)):

            bui = bui_values[i]

            if not np.isfinite(bui):
                bui = np.nan

            all_results.append({
                "id": sub_df["id"].iloc[i],
                "lat": sub_df["lat"].iloc[i],
                "lon": sub_df["lon"].iloc[i],
                "province": province,
                "time": t_val,
                "upwelling_index_ekman": bui,
                "nearest_ekman_distance_km": distance_km[i]
            })


        # if province == 14:
        #     print(f"\n-- DEBUG: Checking Ekman data for province 14 at time {t_val.strftime('%Y-%m')}")

        #     ektrx_vals = ektrx.interp(
        #         latitude=("points", sub_df["lat"].values),
        #         longitude=("points", sub_df["lon"].values),
        #         method='nearest'
        #     ).values

        #     ektry_vals = ektry.interp(
        #         latitude=("points", sub_df["lat"].values),
        #         longitude=("points", sub_df["lon"].values),
        #         method='nearest'
        #     ).values

        #     # Check for missing values
        #     num_nans_x = np.isnan(ektrx_vals).sum()
        #     num_nans_y = np.isnan(ektry_vals).sum()
        #     print(f"  Number of NaNs in ektrx: {num_nans_x} / {len(ektrx_vals)}")
        #     print(f"  Number of NaNs in ektry: {num_nans_y} / {len(ektry_vals)}")

        #     if num_nans_x == len(ektrx_vals) or num_nans_y == len(ektry_vals):
        #         print(f"All Ekman values missing for province 14 in {t_val.strftime('%Y-%m')}")
                

#   5: Save Point-Level Results (Append) 
ekman_bui_df = pd.DataFrame(all_results)

# Quality control
ekman_bui_df["upwelling_index_ekman"] = pd.to_numeric(
    ekman_bui_df["upwelling_index_ekman"],
    errors="coerce"
)

ekman_bui_df["upwelling_index_ekman"] = (
    ekman_bui_df["upwelling_index_ekman"]
    .replace([np.inf, -np.inf], np.nan)
)

valid_bui = ekman_bui_df[
    "upwelling_index_ekman"
].dropna()

print("\n===== BUI QUALITY CONTROL ====")

print("Total coastal observations:", len(ekman_bui_df))
print("Valid BUI observations:", len(valid_bui))
print("Missing BUI observations:",
      ekman_bui_df["upwelling_index_ekman"].isna().sum())

if len(valid_bui) > 0:
    print("Minimum BUI:", valid_bui.min())
    print("Maximum BUI:", valid_bui.max())
    print("Mean BUI:", valid_bui.mean())

print("========\n")

# Check if file exists
file_exists = os.path.isfile(output_csv)

ekman_bui_df.to_csv(
    output_csv,
    mode='a' if file_exists else 'w',
    header=not file_exists,
    index=False
)
print(f"\n Ekman-based upwelling index values {'appended to' if file_exists else 'saved to'}: {output_csv}")

#   6: Save Province-Level Monthly Averages (Append) 
#province_avg = ekman_bui_df.groupby(["province", "time"])["upwelling_index_ekman"].mean().reset_index()

province_avg = (
    ekman_bui_df
    .groupby(["province", "time"])
    .agg(
        upwelling_index_ekman=("upwelling_index_ekman", "mean"),
        n_valid=("upwelling_index_ekman", "count"),
        n_total=("upwelling_index_ekman", "size")
    )
    .reset_index()
)

province_avg["n_missing"] = (
    province_avg["n_total"] - province_avg["n_valid"]
)

avg_file_exists = os.path.isfile(avg_output_csv)

province_avg.to_csv(
    avg_output_csv,

    mode='a' if avg_file_exists else 'w',
    header=not avg_file_exists,
    index=False
)
print(f"Province-averaged upwelling index {'appended to' if avg_file_exists else 'saved to'}: {avg_output_csv}")
