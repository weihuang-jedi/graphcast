#!/usr/bin/env python3
"""
Cartesian 3D Spherical Regridder for Terrain-Following Height Coordinates.
Regrids M6 icosahedral mesh forecasts (40,962 nodes) directly onto regular 0.25-degree 
Lat-Lon grids level-by-level along terrain-following geometric height levels (eta / target_level).
"""

import os
import glob
import argparse
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree


def extract_node_coords(ds):
    """Extracts 1D node latitude and longitude arrays from an icosahedral dataset."""
    for lat_key in ["latitude", "lat", "lat_icosahedral", "lats"]:
        if lat_key in ds.coords or lat_key in ds.data_vars:
            lats = np.ravel(np.asarray(ds[lat_key].values, dtype=np.float32))
            break
    else:
        raise KeyError("Could not find latitude variable in icosahedral dataset.")

    for lon_key in ["longitude", "lon", "lon_icosahedral", "lons"]:
        if lon_key in ds.coords or lon_key in ds.data_vars:
            lons = np.ravel(np.asarray(ds[lon_key].values, dtype=np.float32))
            break
    else:
        raise KeyError("Could not find longitude variable in icosahedral dataset.")

    if np.abs(lats).max() <= 1.58:
        lats = np.degrees(lats)
    if np.abs(lons).max() <= 3.15 and np.min(lons) < 0:
        lons = np.degrees(lons)

    lons = np.mod(lons, 360.0)
    return lats, lons


def extract_target_grid(ds):
    """Extracts 1D latitude and longitude coordinate axes for the output regular grid."""
    for lat_key in ["latitude", "lat"]:
        if lat_key in ds.coords or lat_key in ds.data_vars:
            lats = np.unique(np.ravel(ds[lat_key].values))
            break
    else:
        raise KeyError("Could not find latitude variable in target grid dataset.")

    for lon_key in ["longitude", "lon"]:
        if lon_key in ds.coords or lon_key in ds.data_vars:
            lons = np.unique(np.ravel(ds[lon_key].values))
            break
    else:
        raise KeyError("Could not find longitude variable in target grid dataset.")

    lons = np.mod(lons, 360.0)
    return np.sort(lats)[::-1], np.sort(lons)  # Ensure decreasing latitude (90 -> -90)


def regrid_forecasts(fcst_dir, truth_ref, grid_ref, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load Icosahedral Mesh Node Coordinates
    print(f"[REGRID] Loading icosahedral mesh reference: {grid_ref}")
    ds_grid = xr.open_dataset(grid_ref)
    node_lats, node_lons = extract_node_coords(ds_grid)
    num_nodes = len(node_lats)
    ds_grid.close()

    print(f"[REGRID] Found {num_nodes} icosahedral mesh nodes.")

    rad_lat = np.radians(node_lats)
    rad_lon = np.radians(node_lons)
    x_nodes = np.cos(rad_lat) * np.cos(rad_lon)
    y_nodes = np.cos(rad_lat) * np.sin(rad_lon)
    z_nodes = np.sin(rad_lat)
    node_pts = np.column_stack((x_nodes, y_nodes, z_nodes))

    tree = cKDTree(node_pts)

    # 2. Extract Target Regular Grid Metadata & Vertical Coordinate Attributes
    print(f"[REGRID] Loading target grid reference: {truth_ref}")
    ds_truth = xr.open_dataset(truth_ref)
    target_lats, target_lons = extract_target_grid(ds_truth)

    eta = ds_truth['eta'].values if 'eta' in ds_truth else np.linspace(1.0, 0.0, 32)
    target_level = ds_truth['target_level'].values if 'target_level' in ds_truth else np.arange(32)
    levels = ds_truth['level'].values if 'level' in ds_truth else np.arange(32)
    num_levels = len(levels)
    ds_truth.close()

    print(f"[REGRID] Target grid resolution: {len(target_lats)} x {len(target_lons)} across {num_levels} terrain height levels")

    lon_grid, lat_grid = np.meshgrid(target_lons, target_lats)
    t_rad_lat = np.radians(lat_grid.ravel())
    t_rad_lon = np.radians(lon_grid.ravel())

    target_pts = np.column_stack(
        (
            np.cos(t_rad_lat) * np.cos(t_rad_lon),
            np.cos(t_rad_lat) * np.sin(t_rad_lon),
            np.sin(t_rad_lat),
        )
    )

    # 3. Query 4 Nearest Neighbors with Gaussian Distance Weighting (M6 sigma = 0.0125)
    dists, indices = tree.query(target_pts, k=4)
    sigma = 0.0125
    weights = np.exp(-(dists**2) / (2 * sigma**2))
    weights /= weights.sum(axis=-1, keepdims=True)

    # 4. Process Forecast Files
    fcst_files = sorted(glob.glob(os.path.join(fcst_dir, "*.nc")))
    print(f"[REGRID] Regridding {len(fcst_files)} forecast files to terrain-following lat-lon grid...")

    for fpath in fcst_files:
        fname = os.path.basename(fpath)
        ds_in = xr.open_dataset(fpath)

        out_vars = {}
        for var in ["P", "Q", "T", "U", "V", "W", "h_icosahedral"]:
            var_key = var if var in ds_in else var.lower()
            if var_key in ds_in:
                val = np.squeeze(ds_in[var_key].values)  # (32, num_nodes)

                # Convert Pressure to hPa if saved in Pa (~100,000)
                if var.upper() == "P" and np.nanmean(val) > 2000.0:
                    val = val / 100.0

                # Spatial Gaussian 4-NN regridding: (32, 40962) -> (32, n_lat, n_lon)
                regrid_val = np.sum(
                    val[:, indices] * weights[None, :, :], axis=-1
                )
                
                out_name_var = "h" if var == "h_icosahedral" else var.lower()
                out_vars[out_name_var] = (
                    ["level", "latitude", "longitude"],
                    regrid_val.reshape(
                        num_levels, len(target_lats), len(target_lons)
                    ),
                )

        ds_out = xr.Dataset(
            data_vars=out_vars,
            coords={
                "level": levels,
                "latitude": target_lats,
                "longitude": target_lons,
                "eta": ("level", eta),
                "target_level": ("level", target_level),
            },
            attrs={
                "title": f"M6 Regridded Forecast on Terrain-Following Height Grid",
                "source": "AIDA GNN Weather Model",
            },
        )

        ds_out.to_netcdf(os.path.join(out_dir, f"reconstructed_{fname}"))
        ds_in.close()

    print(f"[SUCCESS] Regridding complete. Files saved to '{out_dir}'.")


def main():
    parser = argparse.ArgumentParser(
        description="Regrid M6 icosahedral mesh forecasts to terrain-following Lat-Lon grid."
    )
    parser.add_argument(
        "--fcst_dir", required=True, help="Directory containing raw forecast .nc files"
    )
    parser.add_argument(
        "--truth_ref",
        required=True,
        help="Regular Terrain-Following Lat-Lon target reference NetCDF file",
    )
    parser.add_argument(
        "--grid_ref",
        required=True,
        help="Icosahedral M6 mesh topology reference NetCDF file",
    )
    parser.add_argument(
        "--out_dir", required=True, help="Output directory for reconstructed NetCDFs"
    )
    args = parser.parse_args()

    regrid_forecasts(args.fcst_dir, args.truth_ref, args.grid_ref, args.out_dir)


if __name__ == "__main__":
    main()
