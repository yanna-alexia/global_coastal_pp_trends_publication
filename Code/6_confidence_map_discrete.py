"""
author: yfi

This script creates a global map showing the confidence scores of primary production (PP) trends across Longhurst provinces. 
It matches province IDs in a NetCDF geographical region map with scores from a CSV file, colours each province according to 
its score using a diverging colour scale, adds province acronyms and boundary contours, and saves the resulting map as a 
high-resolution PNG.

"""

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import os
import matplotlib.patheffects as PathEffects

# File Paths
nc_path = "/users/rsg/yfi/Documents/FOCUS/longhurst/longhurst_coastal_regions-ONLY.nc"
csv_path = "/users/rsg-new/yfi/Documents/FOCUS/June_2025/PP/Plots/pp_trend_scores_summary2.csv"
output_path = "/users/rsg-new/yfi/Documents/FOCUS/June_2025/PP/Plots/pp_confidence_score_map_flip_label2_discrete.png"

# Helper: Compute grid edges from centers
def edges_from_centers(arr):
    edges = np.zeros(len(arr) + 1)
    edges[1:-1] = (arr[:-1] + arr[1:]) / 2
    edges[0] = arr[0] - (edges[1] - arr[0])
    edges[-1] = arr[-1] + (arr[-1] - edges[-2])
    return edges

# Load province region map
ds = xr.open_dataset(nc_path)
region = ds["Region"].values

# Flip and align coordinates
lat = np.linspace(90, -90, ds.dims['lat'])[::-1]
lon = np.linspace(0, 360, ds.dims['lon'], endpoint=False)
lon = ((lon + 180) % 360) - 180  # shift to -180 to 180
sort_idx = np.argsort(lon)
lon = lon[sort_idx]
region = np.roll(region, shift=(region.shape[1] // 2 - np.where(lon == 0)[0][0]), axis=1)
lon = np.roll(lon, shift=(region.shape[1] // 2 - np.where(lon == 0)[0][0]))

# Grids for plotting
lon_grid, lat_grid = np.meshgrid(lon, lat)
lon_edges = edges_from_centers(lon)
lat_edges = edges_from_centers(lat)
lon_edges_grid, lat_edges_grid = np.meshgrid(lon_edges, lat_edges)

# Load trend confidence scores
df = pd.read_csv(csv_path)
score_map = np.full(region.shape, np.nan)

# Fill score map by province ID
for _, row in df.iterrows():
    prov_id = row["provnum"]
    score = row["total_score"]
    mask = region == prov_id
    score_map[mask] = score

# Colour normalization
vmax = 6
vmin = -6
norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
cmap = plt.get_cmap('Spectral_r') #('PRGn')  #https://matplotlib.org/stable/users/explain/colors/colormaps.html gist_rainbow rainbow

# Plotting
fig, ax = plt.subplots(figsize=(14, 7))
#ax.set_title("Confidence Score of Trends by Province", fontsize=20)

img = ax.pcolormesh(
    lon_edges_grid, lat_edges_grid, score_map,
    cmap=cmap, norm=norm, shading='auto'
)

# Draw black contours for province boundaries
ax.contour(lon, lat, region, levels=np.unique(region), colors='k', linewidths=0.5, linestyles='solid', zorder=30)

# Province acronyms
for _, row in df.iterrows():
    prov_id = row["provnum"]
    label = row["Acronym"]
    mask = region == prov_id
    if not np.any(mask):
        continue

    # Compute centroid
    x = lon_grid[mask].mean()
    y = lat_grid[mask].mean()

    # Manual adjustments  province label
    if prov_id == 35:  # INDE
        y += 5.0
    elif prov_id == 36:  # INDW
        y -= 1.0
    elif prov_id == 72:  # NEWZ
        x += 79.0


    # Plot text
    txt = ax.text(
        x, y,
        label,
        fontsize=11,
        color='black',
        ha='center', va='center',
        zorder=50
    )

    
    # Add white outline
    txt.set_path_effects([
        PathEffects.withStroke(linewidth=2, foreground='white')
    ])

# Axes settings
ax.set_xlim(lon.min(), lon.max())
ax.set_ylim(lat.min(), lat.max())
ax.set_xlabel("Longitude", fontsize=16)
ax.set_ylabel("Latitude", fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)

# Colorbar
# cbar = plt.colorbar(img, ax=ax, orientation='vertical', pad=0.02)
# cbar.set_label("Score", fontsize=14)
# cbar.ax.tick_params(labelsize=14)
# cbar.set_ticks(np.arange(vmin, vmax + 1, 1))

# Colorbar discrete
cbar = plt.colorbar(
    img,
    ax=ax,
    orientation='vertical',
    pad=0.02,
    boundaries=np.arange(-6.5, 7.5, 1),  # category edges
    ticks=np.arange(-6, 7, 1),            # category labels
    spacing='uniform',
    drawedges=True                        # draw lines between categories
)

cbar.set_label("Score", fontsize=14)
cbar.ax.tick_params(labelsize=14)

# Save
plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Confidence score map saved to: {output_path}")
