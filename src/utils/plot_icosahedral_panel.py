#!/usr/bin/env python3
"""
plot_icosahedral_panel.py
-------------------------
Generates a 6-panel diagnostic visualization (T, p, q, u, v, w) on a global
Robinson projection for a specified vertical level index from an unstructured
icosahedral NetCDF dataset without Date Line / Pacific seam triangulation stripes.
"""

import argparse
import os
import sys
import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def plot_icosahedral_level_panel(nc_file: str, level_idx: int = 0, output_png: str = None, show = False):
    if not os.path.exists(nc_file):
        raise FileNotFoundError(f"[ERROR] Input NetCDF file not found: '{nc_file}'")

    print(f"[PLOTTER] Opening NetCDF dataset: {nc_file}")
    ds = xr.open_dataset(nc_file)

    num_levels = ds.sizes.get('level', 32)
    if level_idx < 0 or level_idx >= num_levels:
        raise ValueError(f"[ERROR] Requested level_idx {level_idx} out of range [0, {num_levels - 1}].")

    # Extract coordinates & face connectivity
    lons = ds['longitude'].values      # [node]
    lats = ds['latitude'].values       # [node]
    faces = ds['face_nodes'].values    # [face, 3]

    # Map longitudes from [0, 360] to [-180, 180] for Cartopy
    lons_plot = np.where(lons > 180.0, lons - 360.0, lons)

    # -------------------------------------------------------------------------
    # SEAM STRIPE ELIMINATION: Filter out triangles crossing the Date Line
    # -------------------------------------------------------------------------
    face_lons = lons_plot[faces]  # Shape: [num_faces, 3]
    lon_diffs = np.max(face_lons, axis=1) - np.min(face_lons, axis=1)
    
    # Any triangle spanning more than 180 degrees crosses the Date Line seam
    cross_dateline_mask = lon_diffs > 180.0
    clean_faces = faces[~cross_dateline_mask]

    # Build triangulation with clean faces only
    triangulation = mtri.Triangulation(lons_plot, lats, triangles=clean_faces)

    # -------------------------------------------------------------------------
    # Read dynamic fields for requested vertical level
    # -------------------------------------------------------------------------
    # 1. Temperature T (K) from ln_t
    ln_t = ds['ln_t_icosahedral'].isel(level=level_idx).values
    t_k = np.exp(ln_t)

    # 2. Pressure p (hPa) from ln_p
    ln_p = ds['ln_p_icosahedral'].isel(level=level_idx).values
    p_hpa = np.exp(ln_p) / 100.0

    # 3. Specific Humidity q (g/kg)
    q_kgkg = ds['q_icosahedral'].isel(level=level_idx).values
    q_gkg = q_kgkg * 1000.0

    # 4. Zonal Wind u (m/s)
    u_ms = ds['u_icosahedral'].isel(level=level_idx).values

    # 5. Meridional Wind v (m/s)
    v_ms = ds['v_icosahedral'].isel(level=level_idx).values

    # 6. Vertical Velocity w (Pa/s)
    w_pas = ds['w_icosahedral'].isel(level=level_idx).values

    # Height metadata
    if 'h_icosahedral' in ds:
        h_mean = float(np.nanmean(ds['h_icosahedral'].isel(level=level_idx).values))
        level_str = f"Level {level_idx + 1}/{num_levels} (~{h_mean:.0f} m ASL)"
    else:
        level_str = f"Level {level_idx + 1}/{num_levels}"

    print(f"[PLOTTER] Rendering 6-panel figure for {level_str}...")

    # Set up Subplot Grid
    proj = ccrs.Robinson(central_longitude=0.0)
    data_proj = ccrs.PlateCarree()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), subplot_kw={'projection': proj})
    axes = axes.flatten()

    panels = [
        {'data': t_k,   'title': 'Temperature (T)',     'units': 'K',    'cmap': 'coolwarm',  'ax_idx': 0},
        {'data': p_hpa, 'title': 'Pressure (p)',        'units': 'hPa',  'cmap': 'viridis_r', 'ax_idx': 1},
        {'data': q_gkg, 'title': 'Specific Humidity (q)','units': 'g/kg', 'cmap': 'Blues',     'ax_idx': 2},
        {'data': u_ms,  'title': 'Zonal Wind (u)',       'units': 'm/s',  'cmap': 'RdBu_r',    'ax_idx': 3},
        {'data': v_ms,  'title': 'Meridional Wind (v)',  'units': 'm/s',  'cmap': 'RdBu_r',    'ax_idx': 4},
        {'data': w_pas, 'title': 'Vertical Velocity (w)','units': 'Pa/s', 'cmap': 'seismic',   'ax_idx': 5},
    ]

    for p in panels:
        ax = axes[p['ax_idx']]
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', alpha=0.7)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':', edgecolor='gray')
        ax.gridlines(draw_labels=False, linewidth=0.3, color='gray', alpha=0.5, linestyle='--')

        # Render filled triangles using cleaned mesh topology
        cntr = ax.tripcolor(
            triangulation,
            p['data'],
            cmap=p['cmap'],
            shading='flat',
            transform=data_proj
        )

        ax.set_title(f"{p['title']} [{p['units']}]", fontsize=12, fontweight='bold', pad=8)
        cbar = fig.colorbar(cntr, ax=ax, orientation='horizontal', pad=0.05, shrink=0.85, aspect=25)
        cbar.ax.tick_params(labelsize=9)

    file_title = os.path.basename(nc_file)
    fig.suptitle(f"AIDA Icosahedral Mesh Meteorological Diagnostic Panel (Seam-Filtered)\nFile: {file_title} | {level_str}",
                 fontsize=15, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_png is None:
        output_png = f"panel_level_{level_idx + 1:02d}_{os.path.splitext(file_title)[0]}.png"

    plt.savefig(output_png, dpi=250, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    ds.close()

    print(f"[SUCCESS] Multi-panel diagnostic plot generated and saved to: '{output_png}'\n")


def main():
    parser = argparse.ArgumentParser(description="Plot 6-panel (T, p, q, u, v, w) Diagnostic Figure from Icosahedral NetCDF")
    parser.add_argument("-i", "--input", required=True, help="Path to input icosahedral NetCDF file")
    parser.add_argument("-l", "--level", type=int, default=0, help="Vertical level index (0 to 31, default: 0)")
    parser.add_argument("-o", "--output", help="Destination path for output PNG image")
    parser.add_argument("-s", "--show", action="store_true", help="Display plot interactively")

    args = parser.parse_args()

    plot_icosahedral_level_panel(
        nc_file=args.input,
        level_idx=args.level,
        output_png=args.output,
        show=args.show
    )


if __name__ == "__main__":
    main()
