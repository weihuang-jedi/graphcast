#!/usr/bin/env python3
"""
Cartesian 3D Spherical Regridder with Gaussian Inverse-Distance Weighting.
Regridding icosahedral mesh forecasts onto standard 1-degree Lat-Lon grids.
"""

import os
import glob
import argparse
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

def regrid_forecasts(fcst_dir, truth_ref, grid_ref, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Load Icosahedral Mesh Coordinates
    ds_grid = xr.open_dataset(grid_ref)
    node_lats = ds_grid['latitude'].values if 'latitude' in ds_grid else ds_grid['lat'].values
    node_lons = ds_grid['longitude'].values if 'longitude' in ds_grid else ds_grid['lon'].values
    ds_grid.close()

    # Convert icosahedral nodes to 3D Cartesian coordinates
    rad_lat = np.radians(node_lats)
    rad_lon = np.radians(node_lons)
    x_nodes = np.cos(rad_lat) * np.cos(rad_lon)
    y_nodes = np.cos(rad_lat) * np.sin(rad_lon)
    z_nodes = np.sin(rad_lat)
    node_pts = np.column_stack((x_nodes, y_nodes, z_nodes))

    tree = cKDTree(node_pts)

    # 2. Target 1-Degree Lat-Lon Grid
    ds_truth = xr.open_dataset(truth_ref)
    target_lats = ds_truth['latitude'].values if 'latitude' in ds_truth else ds_truth['lat'].values
    target_lons = ds_truth['longitude'].values if 'longitude' in ds_truth else ds_truth['lon'].values
    ds_truth.close()

    lon_grid, lat_grid = np.meshgrid(target_lons, target_lats)
    t_rad_lat = np.radians(lat_grid.ravel())
    t_rad_lon = np.radians(lon_grid.ravel())

    target_pts = np.column_stack((
        np.cos(t_rad_lat) * np.cos(t_rad_lon),
        np.cos(t_rad_lat) * np.sin(t_rad_lon),
        np.sin(t_rad_lat)
    ))

    # Query 4 nearest neighbors with Gaussian distance weighting
    dists, indices = tree.query(target_pts, k=4)
    # M4 Mesh Variance: sigma = 0.05
    # weights = np.exp(-(dists**2) / (2 * 0.05**2))
    # weights /= weights.sum(axis=-1, keepdims=True)
    # M6 Mesh Variance: sigma = 0.0125 (tighter Gaussian weighting for 100km mesh)
    weights = np.exp(-(dists**2) / (2 * 0.0125**2))
    weights /= weights.sum(axis=-1, keepdims=True)

    # 3. Process Forecast Files
    fcst_files = sorted(glob.glob(os.path.join(fcst_dir, "*.nc")))
    print(f"Regridding {len(fcst_files)} forecast files using 4-NN Gaussian weighting...")

    for fpath in fcst_files:
        fname = os.path.basename(fpath)
        ds_in = xr.open_dataset(fpath)

        out_vars = {}
        for var in ["P", "Q", "T", "U", "V", "W"]:
            if var in ds_in:
                val = ds_in[var].values # (level, node)
                # Weighted average over 4 nearest neighbors
                regrid_val = np.sum(val[:, indices] * weights[None, :, :], axis=-1)
                out_vars[var.lower()] = (["level", "latitude", "longitude"], regrid_val.reshape(32, len(target_lats), len(target_lons)))

        ds_out = xr.Dataset(
            data_vars=out_vars,
            coords={
                "level": np.arange(32),
                "latitude": target_lats,
                "longitude": target_lons,
            },
            attrs=ds_in.attrs
        )
        ds_out.to_netcdf(os.path.join(out_dir, f"reconstructed_{fname}"))
        ds_in.close()

    print("[SUCCESS] Regridding complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fcst_dir", required=True)
    parser.add_argument("--truth_ref", required=True)
    parser.add_argument("--grid_ref", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    regrid_forecasts(args.fcst_dir, args.truth_ref, args.grid_ref, args.out_dir)
