#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: yfi

This script calculates the annual percentage change in primary production (PP) for each Longhurst province using linear regression, 
then combines these results with trend statistics to assess statistical significance. It produces a global map showing percentage 
changes, with significant trends displayed as filled colours and non-significant trends represented by dots, and labels provinces 
with an asterisk where trends are significant.
"""

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import linregress

# Paths
nc_path = "/path/to/input/"
trend_csv = "/path/to/input/" 
pp_ts_path = "/path/to/input/"
output_path = "/path/to/output/" 

# Helper
def edges_from_centers(arr):
    edges = np.zeros(len(arr) + 1)
    edges[1:-1] = (arr[:-1] + arr[1:]) / 2
    edges[0] = arr[0] - (edges[1] - arr[0])
    edges[-1] = arr[-1] + (arr[-1] - edges[-2])
    return edges

# Load region map
ds = xr.open_dataset(nc_path)
region = ds["Region"].values

lat = np.linspace(90, -90, ds.dims["lat"])[::-1]
lon = np.linspace(0, 360, ds.dims["lon"], endpoint=False)
lon = ((lon + 180) % 360) - 180

sort_idx = np.argsort(lon)
lon = lon[sort_idx]
region = np.roll(
    region,
    shift=(region.shape[1] // 2 - np.where(lon == 0)[0][0]),
    axis=1
)
lon = np.roll(lon, shift=(region.shape[1] // 2 - np.where(lon == 0)[0][0]))

lon_grid, lat_grid = np.meshgrid(lon, lat)
lon_edges = edges_from_centers(lon)
lat_edges = edges_from_centers(lat)
lon_edges_grid, lat_edges_grid = np.meshgrid(lon_edges, lat_edges)

# Load monthly PP time series
pp_df = pd.read_csv(pp_ts_path)
pp_df = pp_df[["year", "month", "provnum", "pp_mask80"]] 
pp_df.rename(columns={"pp_mask80": "pp"}, inplace=True) 

# decimal year
pp_df["time"] = pp_df["year"] + (pp_df["month"] - 0.5) / 12

# Compute mean PP per province
stats = []

for prov, g in pp_df.groupby("provnum"):
    g = g.dropna(subset=["pp"])
    if len(g) < 24:
        continue

    mean_pp = g["pp"].mean()
    slope, _, _, _, _ = linregress(g["time"], g["pp"])

    stats.append({
        "provnum": prov,
        "pp_mean": mean_pp,
        "pp_slope_ts": slope
    })

stats_df = pd.DataFrame(stats)

# Load Santer trend results
trend_df = pd.read_csv(trend_csv)

# Merge
df = trend_df.merge(stats_df, on="provnum", how="left")

# Compute % change per year
df["pp_pct_change_mask80"] = np.where(
    df["pp_mean"] > 0,
    df["pp_slope_mask80"] / df["pp_mean"] * 100,
    np.nan
)

# Optional? safety cap
df["pp_pct_change_mask80"] = df["pp_pct_change_mask80"].clip(-200, 200)

#print(df[["provnum", "pp_mean", "pp_slope_nomask", "pp_pct_change_nomask"]])
#print("Slope min/max:", df["pp_slope_nomask"].min(), df["pp_slope_nomask"].max())
#print("% change min/max:", df["pp_pct_change_nomask"].min(), df["pp_pct_change_nomask"].max())


# Build maps
pct_map = np.full(region.shape, np.nan)
significant_mask = np.full(region.shape, False)

for _, row in df.iterrows():
    prov_id = row["provnum"]
    pct = row["pp_pct_change_mask80"]
    sig = row["p_mask80"] < 0.05

    mask = region == prov_id
    pct_map[mask] = pct
    significant_mask[mask] = sig

# Plot
abs_max = np.nanmax(np.abs(pct_map))
norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
cmap = plt.get_cmap("coolwarm")

fig, ax = plt.subplots(figsize=(14, 7))
#ax.set_title("PP % Change by Province (1998–2022)", fontsize=20)

# Significant = filled
pct_sig = np.where(significant_mask, pct_map, np.nan)
img = ax.pcolormesh(
    lon_edges_grid, lat_edges_grid,
    pct_sig, cmap=cmap, norm=norm, shading="auto"
)

# Non-significant = dots
nonsig = (~significant_mask) & (~np.isnan(pct_map))
ax.scatter(
    lon_grid[nonsig][::60],
    lat_grid[nonsig][::60],
    c=cmap(norm(pct_map[nonsig][::60])),
    s=4, marker="o", edgecolors="none",
    zorder=20
)

# Province boundaries
ax.contour(
    lon, lat, region,
    levels=np.unique(region),
    colors="k", linewidths=0.5, zorder=30
)

ax.set_xlim(lon.min(), lon.max())
ax.set_ylim(lat.min(), lat.max())
ax.set_xlabel("Longitude", fontsize=16)
ax.set_ylabel("Latitude", fontsize=16)
ax.tick_params(labelsize=14)

# Colorbar
cbar = plt.colorbar(img, ax=ax, pad=0.02)
cbar.set_label("Change per year (%)", fontsize=14)
cbar.ax.tick_params(labelsize=14)

# # Province labels (offset + *)
# label_offset = 2.0  # degrees longitude

# for _, row in df.iterrows():
#     prov_id = row["provnum"]
#     label = row["Acronym"]

#     if row["p_nomask"] < 0.05:
#         label = f"{label}*"

#     mask = region == prov_id
#     if not np.any(mask):
#         continue

#     ax.text(
#         lon_grid[mask].mean(), #+ label_offset,
#         lat_grid[mask].mean(),
#         label,
#         fontsize=12,
#         ha="left", va="center",
#         zorder=50,
#         bbox=dict(
#             facecolor="white",
#             edgecolor="none",
#             alpha=0.7,
#             boxstyle="round,pad=0.2"
#         )
#     )

for _, row in df.iterrows():
    prov_id = row["provnum"]
    label = row["Acronym"]

    if row["p_mask80"] < 0.05:
        label = f"{label}*"

    mask = region == prov_id
    if not np.any(mask):
        continue

    # centroid
    x = lon_grid[mask].mean()
    y = lat_grid[mask].mean()

    # manually move specific province labels in a direction
    if prov_id == 35:  # INDE
        y += 5.0  # move 1 degree up
    elif prov_id == 36:  # INDW
        y -= 1.0  # move 1 degree down
    elif prov_id == 72:  # NEWZ
        x += 75.0  # move 2 degrees right

    ax.text(
        x,
        y,
        label,
        fontsize=12,
        ha="left", va="center",
        zorder=50,
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.7,
            boxstyle="round,pad=0.2"
        )
    )


plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Map saved to: {output_path}")
